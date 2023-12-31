import os
import glob
from typing import Tuple
from pprint import pprint

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.modeling.data_loading import DroneWaterSegmentationDataset
from src.modeling.transformation import TransformerConfig
from src.modeling.training import fit, DroneTrainer
from src.modeling.custom_losses import DiceBCELoss
from src.utils import move_to
from src.constants import forest_seg_h, forest_seg_w, incorrect_segm_maps
from src.modeling.smp_doc_model import ForestModel


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

        im_paths = glob.glob(os.path.join(train_path, 'images', '**/**.jpg'), recursive=True)[:20]
        mask_paths = glob.glob(os.path.join(train_path, 'masks', '**/**.jpg'), recursive=True)[:20]

        train_ims, val_ims, train_masks, val_masks = train_test_split(im_paths, mask_paths, test_size=0.1)

        # setup training
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_set = DroneWaterSegmentationDataset(
            im_paths=train_ims,
            mask_paths=train_masks,
            mean=mean,
            std=std,
            fg_color=self.fg_color,
            transform=self.t_train,
        )
        val_set = DroneWaterSegmentationDataset(
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

        # for ims, masks in train_loader:
        #     import numpy as np
        #     sample = ims[0].detach().cpu().numpy()
        #     sample_mask = masks[0].detach().cpu().numpy()
        #     sample_mask = np.moveaxis(sample_mask, 0, -1)
        #     sample = np.moveaxis(sample, 0, -1)
        #     from matplotlib import pyplot as plt
        #     plt.imshow(sample)
        #     plt.show()
        #     exit(0)

        # model = smp.Unet(
        #     self.backbone,
        #     encoder_weights='imagenet',
        #     classes=1,
        #     # activation=None,
        #     activation='sigmoid',
        #     encoder_depth=5,
        #     decoder_channels=[256, 128, 64, 32, 16],
        # )
        model = smp.FPN(
            encoder_name=self.backbone,
            encoder_weights=None,
            classes=1,
            activation='sigmoid',
        )
        model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=1, activation='sigmoid', encoder_depth=5,
                         decoder_channels=[256, 128, 64, 32, 16])

        # model = ForestModel("FPN", "resnet34", in_channels=3, out_classes=1)
        # trainer = pl.Trainer(
        #     # gpus=1,
        #     max_epochs=50,
        # )
        #
        # trainer.fit(
        #     model,
        #     train_dataloaders=train_loader,
        #     val_dataloaders=val_loader,
        # )
        # valid_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)
        # pprint(valid_metrics)

        eta_0 = 0.001
        # max_lr = 1e-3
        epoch = 50
        # weight_decay = 1e-4
        #
        # criterion = nn.CrossEntropyLoss()
        # criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)
        criterion = DiceBCELoss()
        #
        # # optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
        # # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
        # #                                                 steps_per_epoch=len(train_loader))
        #
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=eta_0), ])
        scheduler = None
        #
        trainer = DroneTrainer(
            device=self.device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            model_path='model_softmax.pt'
        )
        trainer.train_model(epoch, model, train_loader, val_loader)
        #
        # inputs = torch.randn(1, 3, forest_seg_h, forest_seg_w)
        # inputs = move_to(inputs, self.device)
        #
        # for layer in model.children():
        #     weights = list(layer.parameters())
        #     try:
        #         layer.set_swish(memory_efficient=False)
        #     except Exception as e:
        #         print(str(e))
        #
        # traced_model = torch.jit.trace(model, inputs)
        # traced_model.save('traced_model.pt')
        #
        # exit(0)
