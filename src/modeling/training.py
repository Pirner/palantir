import time
from tqdm import tqdm

import torch
import numpy as np

from src.modeling.utils import get_lr
from src.modeling.metrics import mIoU, pixel_accuracy


class DroneTrainer:
    def __init__(self, device, optimizer, scheduler, criterion, model_path):
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.model_path = model_path
        self.patch = False

        # runtime data
        self.lrs = []
        self.train_losses = []
        self.test_losses = []
        self.train_iou = []
        self.val_iou = []
        self.train_acc = []
        self.val_acc = []

    def run_training_epoch(self, model, train_loader):
        """
        run a training epoch on the training data loader
        :param model: model to train during the epoch
        :param train_loader: training data loader for the epoch
        :return:
        """
        running_loss = 0
        iou_score = 0
        accuracy = 0

        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data
            if self.patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(self.device)
            mask = mask_tiles.to(self.device)
            # forward
            output = model(image)
            loss = self.criterion(output, mask)
            # evaluation metrics
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            self.optimizer.step()  # update weight
            self.optimizer.zero_grad()  # reset gradient

            # step the learning rate
            self.lrs.append(get_lr(self.optimizer))
            if self.scheduler:
                self.scheduler.step()

            running_loss += loss.item()

        self.train_losses.append(running_loss / len(train_loader))
        self.train_iou.append(iou_score / len(train_loader))
        self.train_acc.append(accuracy / len(train_loader))

    def run_validation_epoch(self, model, val_loader):
        """
        run validation epoch on the model
        :param model: model to validate
        :param val_loader: validation data loader
        :return:
        """
        model.eval()
        test_loss = 0
        test_accuracy = 0
        val_iou_score = 0
        # validation loop
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                # reshape to 9 patches from single image, delete batch size
                image_tiles, mask_tiles = data

                if self.patch:
                    bs, n_tiles, c, h, w = image_tiles.size()

                    image_tiles = image_tiles.view(-1, c, h, w)
                    mask_tiles = mask_tiles.view(-1, h, w)

                image = image_tiles.to(self.device)
                mask = mask_tiles.to(self.device)
                output = model(image)
                # evaluation metrics
                val_iou_score += mIoU(output, mask)
                test_accuracy += pixel_accuracy(output, mask)
                # loss
                loss = self.criterion(output, mask)
                test_loss += loss.item()

            self.test_losses.append(test_loss / len(val_loader))
            self.val_iou.append(val_iou_score / len(val_loader))
            self.val_acc.append(test_accuracy / len(val_loader))

    def train_model(self, epochs, model, train_loader, val_loader, patch=False):
        torch.cuda.empty_cache()

        self.train_losses = []
        self.test_losses = []
        self.val_iou = []
        self.val_acc = []
        self.train_iou = []
        self.train_acc = []
        self.lrs = []
        min_loss = np.inf
        decrease = 1
        not_improve = 0

        model.to(self.device)
        fit_time = time.time()

        for e in range(epochs):
            since = time.time()

            model.train()
            self.run_training_epoch(model, train_loader)
            self.run_validation_epoch(model, val_loader)

            # model saving functionality
            # if min_loss > self.test_losses[-1]:
            #     print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, self.test_losses[-1]))
            #     min_loss = self.test_losses[-1]
            #     decrease += 1
            #     if decrease % 5 == 0:
            #         print('saving model...')
            #         torch.save(model, 'Unet-Mobilenet_v2_mIoU-{:.3f}.pt'.format(self.test_losses[-1]))
            #
            # if self.test_losses[-1] > min_loss:
            #     not_improve += 1
            #     min_loss = self.test_losses[-1]
            #     print(f'Loss Not Decrease for {not_improve} time')
            #     if not_improve == 7:
            #         print('Loss not decrease for 7 times, Stop Training')
            #         break

            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.3f}..".format(self.train_losses[-1]),
                  "Val Loss: {:.3f}..".format(self.test_losses[-1]),
                  "Train mIoU:{:.3f}..".format(self.train_iou[-1]),
                  "Val mIoU: {:.3f}..".format(self.val_iou[-1]),
                  "Train Acc:{:.3f}..".format(self.train_acc[-1]),
                  "Val Acc:{:.3f}..".format(self.val_acc[-1]),
                  "Time: {:.2f}m".format((time.time() - since) / 60))
            torch.save(model.state_dict(), self.model_path)


def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, device='cpu', patch=False):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward
            output = model(image)
            loss = criterion(output, mask)
            # evaluation metrics
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1, c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)
                    # evaluation metrics
                    val_iou_score += mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    # loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            # calculate mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))

            if min_loss > (test_loss / len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss / len(val_loader))))
                min_loss = (test_loss / len(val_loader))
                decrease += 1
                if decrease % 5 == 0:
                    print('saving model...')
                    torch.save(model, 'Unet-Mobilenet_v2_mIoU-{:.3f}.pt'.format(val_iou_score / len(val_loader)))

            if (test_loss / len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss / len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 7:
                    print('Loss not decrease for 7 times, Stop Training')
                    break

            # iou
            val_iou.append(val_iou_score / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))
            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                  "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                  "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history



import time
import os

import torch
import pandas as pd
from tqdm import tqdm
import numpy as np


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def move_to(obj, device):
    """
    move the object to the desired device (cpu, cuda)
    :param obj: object to move
    :param device: destination to transport to
    :return:
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to(list(obj), device))
    elif isinstance(obj, set):
        return set(move_to(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[move_to(key, device)] = move_to(value, device)
        return to_ret
    else:
        return obj


class ForestTrainer:
    def __init__(self, im_h: int, im_w: int, run_path: str):
        """
        void trainer class to train a model for void detection
        :param im_h: height of input image
        :param im_w: width of input image
        :param run_path: directory where we store all the run data
        """
        self.im_h = im_h
        self.im_w = im_w
        self.run_path = run_path

    def run_epoch(self, model, optimizer, data_loader, loss_func, device, results, score_funcs, prefix="", desc=None):
        """
        run a single epoch for training
        :param model: model to train
        :param optimizer: optimizer to optimize with
        :param data_loader: data loader to run the epoch against
        :param loss_func: loss function to perform training
        :param device: device to train on
        :param results: result data frame
        :param score_funcs: score functions which measure us (metrics)
        :param prefix: prefix for the namings
        :param desc: ??
        :return:
        """
        running_loss = []
        y_true = []
        y_pred = []
        start = time.time()
        for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
            # Move the batch to the device we are using.
            inputs = move_to(inputs, device)
            labels = move_to(labels, device)

            y_hat = model(inputs)  # this just computed f_Θ(x(i))
            # Compute loss.
            loss = loss_func(y_hat, labels)

            if model.training:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Now we are just grabbing some information we would like to have
            running_loss.append(loss.item())

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

        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(int)

        y_true = np.asarray(y_true)
        y_true = y_true.astype(int)

        y_pred = torch.from_numpy(y_pred)
        y_true = torch.from_numpy(y_true)

        results[prefix + " loss"].append(np.mean(running_loss))
        for name, score_func in score_funcs.items():
            try:
                results[prefix + " " + name].append(score_func(y_pred, y_true).detach().cpu().numpy())
            except Exception as e:
                results[prefix + " " + name].append(float("NaN"))
                print('failed on metrich {0} with {1}'.format(name, str(e)))
        return end - start  # time spent on epoch

    def train_network(self, model, loss_func, train_loader, val_loader=None, test_loader=None,
                      score_funcs=None, epochs=50,
                      device="cpu", checkpoint_file=None, optimizer=None, lr_schedule=None, model_name='model'):
        """
        train an entire network
        :param model: model to train the network
        :param loss_func: loss function for training
        :param train_loader: training data loader
        :param val_loader: validation data loader
        :param test_loader: test data loader
        :param score_funcs: score functions
        :param epochs: epochs to run
        :param device: device to perform computations on
        :param checkpoint_file: filepath for the checkpoint
        :param optimizer: optimizer for optimization
        :param lr_schedule: learning rate schedule
        :return:
        """
        to_track = ["epoch", "total time", "train loss", "learning_rate"]
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
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters())

        # Place the model on the correct compute resource (CPU or GPU)
        model.to(device)
        for epoch in range(epochs):
            print('training epoch {0}/{1}'.format(epoch, epochs))
            model = model.train()  # Put our model in training mode

            total_train_time += self.run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs,
                                               prefix="train", desc="Training")

            results["total time"].append(total_train_time)
            results["epoch"].append(epoch)
            results["learning_rate"].append(get_lr(optimizer))

            if test_loader is not None:
                model = model.eval()
                with torch.no_grad():
                    self.run_epoch(model, optimizer, test_loader, loss_func, device, results, score_funcs,
                                   prefix="test", desc="Testing")

            # In PyTorch, the convention is to update the learning rate after every epoch
            if lr_schedule is not None:
                if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_schedule.step(results["val loss"][-1])
                else:
                    lr_schedule.step()

            if checkpoint_file is not None:
                model_path = os.path.join(checkpoint_file, '{0}.pth'.format(model_name))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'results': results
                }, model_path)

        # store traced model
        inputs = torch.randn(1, 3, self.im_h, self.im_w)
        inputs = move_to(inputs, device)

        for layer in model.children():
            weights = list(layer.parameters())
            try:
                layer.set_swish(memory_efficient=False)
            except Exception as e:
                print(str(e))

        traced_model = torch.jit.trace(model, inputs)
        traced_model.save(os.path.join(checkpoint_file, '{0}_traced.pt'.format(model_name)))

        return pd.DataFrame.from_dict(results)
