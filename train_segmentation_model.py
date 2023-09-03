import os
import glob

import cv2
import pandas as pd

import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch
from src.vision.mask_utils import PalantirMaskUtils
from src.modeling.data_loading import AerialSegmentationSemanticDataset
from src.modeling.augmentations import get_training_augmentation, get_validation_augmentation, get_preprocessing
import segmentation_models_pytorch as smp
from src.modeling.trainer import train_network

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['car']
ACTIVATION = 'softmax2d'  # could be None for logits or 'softmax2d' for multiclass segmentation
# ACTIVATION = 'softmax'
DEVICE = 'cuda'


def main():
    im_w = 640
    im_h = 448
    dataset_path = r'C:\data\drone_aerial_segmentation\archive\preprocessed_data'
    mask_paths = glob.glob(os.path.join(dataset_path, '**/*mask.png'), recursive=True)
    im_paths = glob.glob(os.path.join(dataset_path, '**/*im.png'), recursive=True)
    assert len(im_paths) == len(mask_paths)
    print('[INFO] found {0} data points'.format(len(im_paths)))
    df = pd.read_csv(os.path.join(r'C:\data\drone_aerial_segmentation\archive', 'colormaps.csv'), sep=';')
    print(torch.cuda.is_available())

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=24,
        activation=ACTIVATION,
    )
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=24,
        activation=ACTIVATION,
    )

    train_transform = A.Compose(
        [
            A.Resize(im_h, im_w),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # preprocessing_fn = None

    train_dataset = AerialSegmentationSemanticDataset(
        im_paths,
        mask_paths,
        # augmentation=get_training_augmentation(),
        augmentation=train_transform,
        # preprocessing=get_preprocessing(preprocessing_fn),
        preprocessing=None,
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)

    loss_func = smp.losses.DiceLoss('multiclass')
    loss_fn = nn.CrossEntropyLoss()

    jaccard = torchmetrics.JaccardIndex(num_classes=24, task='multiclass')

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])
    results = train_network(
        model,
        loss_fn,
        train_loader,
        test_loader=None,
        epochs=5,
        optimizer=optimizer,
        lr_schedule=None,
        score_funcs={'jaccard': jaccard},
        # device='cpu'
        device='cuda'
    )

    results.to_csv('training_data.csv')


if __name__ == '__main__':
    main()
