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
from src.constants import forest_seg_h, forest_seg_w


def main():
    dataset_root_path = r'C:\data\palantir\flooded_dataset\dataset'
    train_dataset_src = os.path.join(dataset_root_path, 'train')
    val_dataset_src = os.path.join(dataset_root_path, 'val')

    im_h, im_w = forest_seg_h, forest_seg_w
    batch_size = 2
    epochs = 3

    train_im_paths = glob.glob(os.path.join(train_dataset_src, '**/*train-org-img*/**.jpg'), recursive=True)
    train_mask_paths = glob.glob(os.path.join(train_dataset_src, '**/*train-label-img*/**.png'), recursive=True)

    assert len(train_im_paths) == len(train_mask_paths)

    val_im_paths = glob.glob(os.path.join(val_dataset_src, '**/*val-org-img*/**.jpg'), recursive=True)
    val_mask_paths = glob.glob(os.path.join(val_dataset_src, '**/*val-label-img*/**.png'), recursive=True)

    assert len(val_im_paths) == len(val_mask_paths)

    # im_paths = glob.glob(os.path.join(dataset_src, '**/*sat*.jpg'), recursive=True)
    # mask_paths = glob.glob(os.path.join(dataset_src, '**/*mask*.png'), recursive=True)
    t_train = TransformerConfig.get_train_transforms()

    # train_ims, val_ims, train_masks, val_masks = train_test_split(im_paths, mask_paths, test_size=0.1)

    # setup training
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_set = BinarySatelliteDataset(
        im_paths=train_im_paths,
        mask_paths=train_mask_paths,
        mean=mean,
        std=std,
        transform=t_train,
    )
    val_set = BinarySatelliteDataset(
        im_paths=val_im_paths,
        mask_paths=val_mask_paths,
        mean=mean,
        std=std,
        transform=t_train,
    )
    # dataloader

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

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

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=eta_0),
    ])

    trainer = ForestTrainer(im_h=im_h, im_w=im_w, run_path='')
    results = trainer.train_network(
        model,
        loss_fn,
        train_loader,
        test_loader=val_loader,
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
