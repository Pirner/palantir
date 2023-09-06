import os
import glob

import cv2
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch
import torchvision

from src.vision.mask_utils import PalantirMaskUtils
from src.modeling.data_loading import AerialSegmentationSemanticDataset, SegDatasetInMemory
from src.modeling.augmentations import get_training_augmentation, get_validation_augmentation, get_preprocessing
import segmentation_models_pytorch as smp
from src.modeling.trainer import train_network, SegmentationTrainer
from src.modeling.losses import DiceBCELoss

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['car']
ACTIVATION = 'softmax2d'  # could be None for logits or 'softmax2d' for multiclass segmentation
# ACTIVATION = 'softmax'
DEVICE = 'cuda'


def create_in_memory_dataset(im_paths, mask_paths, im_w, im_h, transforms):
    """
    create in memory dataset
    :param im_paths: image paths to load
    :param mask_paths: mask paths to load
    :param im_w: image width for loading
    :param im_h: image height for loading
    :param transforms: transformations to apply
    :return:
    """
    im_data = []
    mask_data = []

    for im_p, mask_p in tqdm(zip(im_paths, mask_paths)):
        im = cv2.imread(im_p)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(im, (im_w, im_h))
        im_data.append(im_resized)

        mask = cv2.imread(mask_p, 0)
        mask_resized = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_NEAREST_EXACT)
        mask_data.append(mask_resized)
    train_dataset = SegDatasetInMemory(im_data, mask_data, augmentation=transforms)
    return train_dataset


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
    # model = smp.FPN(
    #     encoder_name=ENCODER,
    #     encoder_weights=ENCODER_WEIGHTS,
    #     classes=24,
    #     activation=ACTIVATION,
    # )
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=24,
        activation=ACTIVATION,
    )
    # Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    # Net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes

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
    # for x, y in train_dataset:
    # exit(0)
    # train_dataset = create_in_memory_dataset(
    #     im_paths=im_paths,
    #     mask_paths=mask_paths,
    #     im_h=im_h,
    #     im_w=im_w,
    #     transforms=train_transform,
    # )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    loss_func = smp.losses.DiceLoss('multiclass')
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = DiceBCELoss()

    jaccard = torchmetrics.JaccardIndex(num_classes=24, task='multiclass')

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    trainer = SegmentationTrainer()
    trainer.train_network(
        model,
        loss_fn,
        train_loader,
        test_loader=None,
        epochs=10,
        optimizer=optimizer,
        lr_schedule=None,
        score_funcs={'jaccard': jaccard},
        # device='cpu'
        device='cuda'
    )
    print(trainer.avg_losses)
    # results = train_network(
    #     model,
    #     loss_fn,
    #     train_loader,
    #     test_loader=None,
    #     epochs=5,
    #     optimizer=optimizer,
    #     lr_schedule=None,
    #     score_funcs={'jaccard': jaccard},
    #     # device='cpu'
    #     device='cuda'
    # )
    #
    # results.to_csv('training_data.csv')


if __name__ == '__main__':
    main()
