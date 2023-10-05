import os
import glob

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
import segmentation_models_pytorch as smp
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torchmetrics

from src.modeling.data_loading import BinarySatelliteDataset
from src.modeling.transformation import TransformerConfig
from src.modeling.custom_losses import DiceBCELoss
from src.modeling.training import ForestTrainer


def main():
    dataset_src = r'E:\projects\palantir\forest_segmentation'
    im_h, im_w = 256, 256
    im_paths = glob.glob(os.path.join(dataset_src, 'images', '**/**.jpg'), recursive=True)
    mask_paths = glob.glob(os.path.join(dataset_src, 'masks', '**/**.jpg'), recursive=True)
    t_train = TransformerConfig.get_train_transforms()

    train_ims, val_ims, train_masks, val_masks = train_test_split(im_paths, mask_paths, test_size=0.1)

    # setup training
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_set = BinarySatelliteDataset(
        im_paths=train_ims,
        mask_paths=train_masks,
        mean=mean,
        std=std,
        transform=t_train,
    )
    val_set = BinarySatelliteDataset(
        im_paths=val_ims,
        mask_paths=val_masks,
        mean=mean,
        std=std,
        transform=t_train,
    )
    # dataloader

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=True)

    # model stuff
    ENCODER = 'efficientnet-b0'
    # ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    # ACTIVATION = 'softmax'
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION,
    )

    loss_func = smp.losses.DiceLoss('multiclass')
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.BCELoss()
    loss_fn = DiceBCELoss()

    jaccard = torchmetrics.JaccardIndex(num_classes=1, task='binary')
    eta_min = 0.00001
    eta_0 = 0.0001
    epochs = 50

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=eta_0),
    ])

    trainer = ForestTrainer(im_h=im_h, im_w=im_w, run_path='')
    results = trainer.train_network(
        model,
        loss_fn,
        train_loader,
        test_loader=None,
        epochs=epochs,
        optimizer=optimizer,
        lr_schedule=None,
        score_funcs={'jaccard': jaccard},
        # device='cpu'
        device='cuda',
        checkpoint_file='',
        model_name='baseline'
    )
    results.to_csv('training_data_baseline.csv')


if __name__ == '__main__':
    main()
