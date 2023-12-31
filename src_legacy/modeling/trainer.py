import time

from torchmetrics import JaccardIndex
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np


def moveTo(obj, device):
    """
    obj: the python object to move to a device, or to move its contents to a device
    device: the compute device to move objects to
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    else:
        return obj


class SegmentationTrainer:
    def __init__(self, checkpoint_path: str):
        self.current_epoch = 0
        self.avg_losses = []
        self.checkpoint_path = checkpoint_path
        self.jaccard = JaccardIndex(task="multiclass", num_classes=24)
        self.jaccard_scores = []

    def run_epoch(
            self,
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
    ):
        """
        run epoch for the segmentation trainer
        :param model: model to train
        :param train_loader: training data loader
        :param optimizer: optimizer to train with
        :param loss_fn: loss function to train w ith
        :param device: device to train on
        :return:
        """
        running_loss = 0.
        last_loss = 0.
        losses = []
        jaccard_scores = []

        for inputs, labels in tqdm(train_loader):
            # Move the batch to the device we are using.
            inputs = moveTo(inputs, device)
            labels = moveTo(labels, device)

            y_hat = model(inputs)  # this just computed f_Θ(x(i))
            y_hat_argmax = y_hat.argmax(axis=1)
            loss = loss_fn(y_hat, labels)

            if model.training:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # score the model
            jaccard_score = self.jaccard(moveTo(y_hat_argmax, 'cpu'), moveTo(labels, 'cpu'))

            # Gather data and report
            running_loss += loss.item()
            losses.append(loss.item())
            jaccard_scores.append(jaccard_score.item())

        self.jaccard_scores.append(np.mean(jaccard_scores))
        self.avg_losses.append(np.mean(losses))
        return last_loss

    def train_network(
            self,
            model,
            loss_func,
            train_loader,
            val_loader=None,
            test_loader=None,
            score_funcs=None,
            epochs=50,
            device="cpu",
            checkpoint_file=None,
            optimizer=None,
            lr_schedule=None
    ):
        """
        train a network with all the data
        :param model: model to train
        :param loss_func: loss function to apply
        :param train_loader: train data loader
        :param val_loader: validation data loader
        :param test_loader: test data loader
        :param score_funcs: score functions
        :param epochs: epochs to run
        :param device: device to put data on
        :param checkpoint_file: file to save model
        :param optimizer: optimizer to use
        :param lr_schedule: scheduler for learning rate
        :return:
        """
        model.to(device)

        for i, epoch in enumerate(range(epochs)):
            print('\nrunning {0} of {1}\n'.format(i + 1, epochs))
            self.run_epoch(model, train_loader, optimizer, loss_func, device)
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     # 'results': results
            # }, checkpoint_file)
            print(self.avg_losses)
            print(self.jaccard_scores)


def run_epoch(model, optimizer, data_loader, loss_func, device, results, score_funcs, prefix="", desc=None):
    """
    model -- the PyTorch model / "Module" to run for one epoch
    optimizer -- the object that will update the weights of the network
    data_loader -- DataLoader object that returns tuples of (input, label) pairs.
    loss_func -- the loss function that takes in two arguments, the model outputs and the labels, and returns a score
    device -- the compute lodation to perform training
    score_funcs -- a dictionary of scoring functions to use to evalue the performance of the model
    prefix -- a string to pre-fix to any scores placed into the _results_ dictionary.
    desc -- a description to use for the progress bar.
    """
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
        # Move the batch to the device we are using.
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        y_hat = model(inputs)  # this just computed f_Θ(x(i))
        # Compute loss.
        loss = loss_func(y_hat, labels)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Now we are just grabbing some information we would like to have
        # running_loss.append(loss.detach().item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            # moving labels & predictions back to CPU for computing / storing predictions
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            # add to predictions so far
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())
    # end training epoch
    end = time.time()

    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:  # We have a classification problem, convert to labels
        y_pred = np.argmax(y_pred, axis=1)
    # Else, we assume we are working on a regression problem

    results[prefix + " loss"].append(np.mean(running_loss))
    for name, score_func in score_funcs.items():
        try:
            results[prefix + " " + name].append(score_func(y_true, y_pred))
        except:
            results[prefix + " " + name].append(float("NaN"))
    return end - start  # time spent on epoch


def train_network(model, loss_func, train_loader, val_loader=None, test_loader=None, score_funcs=None, epochs=50,
                  device="cpu", checkpoint_file=None, optimizer=None, lr_schedule=None):
    """Train simple neural networks

    Keyword arguments:
    model -- the PyTorch model / "Module" to train
    loss_func -- the loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score
    train_loader -- PyTorch DataLoader object that returns tuples of (input, label) pairs.
    test_loader -- Optional PyTorch DataLoader to evaluate on after every epoch
    score_funcs -- A dictionary of scoring functions to use to evalue the performance of the model
    epochs -- the number of training epochs to perform
    device -- the compute lodation to perform training

    """
    to_track = ["epoch", "total time", "train loss"]
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score)
        if test_loader is not None:
            to_track.append("test " + eval_score)

    total_train_time = 0  # How long have we spent in the training loop?
    results = {}
    # Initialize every item with an empty list
    for item in to_track:
        results[item] = []

    # SGD is Stochastic Gradient Decent.
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    if optimizer == None:
        optimizer = torch.optim.AdamW(model.parameters())

    # Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch"):
        model = model.train()  # Put our model in training mode

        total_train_time += run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs,
                                      prefix="train", desc="Training")

        results["total time"].append(total_train_time)
        results["epoch"].append(epoch)

        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test",
                          desc="Testing")

        # In PyTorch, the convention is to update the learning rate after every epoch
        if lr_schedule is not None:
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val loss"][-1])
            else:
                lr_schedule.step()

        if checkpoint_file is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results': results
            }, checkpoint_file)

    return pd.DataFrame.from_dict(results)
