import os
import glob
from typing import Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.modeling.data_loading import BinarySatelliteDataset
from src.modeling.transformation import TransformerConfig
from src.modeling.training import fit, DroneTrainer
from src.modeling.custom_losses import DiceBCELoss
from src.utils import move_to
from src.constants import forest_seg_h, forest_seg_w, incorrect_segm_maps


class SatelliteImageTrainer:
    def __init__(self, fg_color: Tuple[int, int, int], backbone='mobilenet_v2', batch_size=1):
        """
        trainer for satellite imagery
        :param fg_color: foreground color
        :param batch_size: batch size for training
        :param backbone: backbone for the model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fg_color = fg_color
        self.t_train = TransformerConfig.get_train_transforms()
        self.batch_size = batch_size
        self.backbone = backbone

    def train_model(self, train_path: str):
        """
        path for the training data - stored
        :param train_path:
        :return:
        """

        im_paths = glob.glob(os.path.join(train_path, 'images', '**/**.jpg'), recursive=True)
        mask_paths = glob.glob(os.path.join(train_path, 'masks', '**/**.jpg'), recursive=True)

        train_ims, val_ims, train_masks, val_masks = train_test_split(im_paths, mask_paths, test_size=0.1)

        # setup training
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_set = BinarySatelliteDataset(
            im_paths=train_ims,
            mask_paths=train_masks,
            mean=mean,
            std=std,
            fg_color=self.fg_color,
            transform=self.t_train,
        )
        val_set = BinarySatelliteDataset(
            im_paths=val_ims,
            mask_paths=val_masks,
            mean=mean,
            std=std,
            fg_color=self.fg_color,
            transform=self.t_train,
        )
        # dataloader

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)

        model = smp.Unet(
            self.backbone,
            encoder_weights='imagenet',
            classes=1,
            # activation=None,
            activation='sigmoid',
            encoder_depth=5,
            decoder_channels=[256, 128, 64, 32, 16],
        )
        model = smp.FPN(
            encoder_name=self.backbone,
            encoder_weights='imagenet',
            classes=1,
            activation='sigmoid',
        )

        eta_0 = 0.001
        max_lr = 1e-3
        epoch = 3
        weight_decay = 1e-4

        criterion = nn.CrossEntropyLoss()
        criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)
        # criterion = DiceBCELoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                        steps_per_epoch=len(train_loader))

        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=eta_0),
        ])
        scheduler = None

        trainer = DroneTrainer(
            device=self.device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            model_path='model_softmax.pt'
        )
        trainer.train_model(epoch, model, train_loader, val_loader)

        inputs = torch.randn(1, 3, forest_seg_h, forest_seg_w)
        inputs = move_to(inputs, self.device)

        for layer in model.children():
            weights = list(layer.parameters())
            try:
                layer.set_swish(memory_efficient=False)
            except Exception as e:
                print(str(e))

        traced_model = torch.jit.trace(model, inputs)
        traced_model.save('traced_model.pt')

        exit(0)
