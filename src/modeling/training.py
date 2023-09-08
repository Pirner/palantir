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
