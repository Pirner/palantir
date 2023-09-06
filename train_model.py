import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

from src.modeling.transformation import TransformerConfig
from src.modeling.data_loading import DroneDataset
from src.modeling.training import fit, DroneTrainer


def create_df(im_data_path):
    name = []
    for dir_name, _, filenames in os.walk(im_data_path):
        for filename in filenames:
            name.append(filename.split('.')[0])

    return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im_data_path = r'C:\data\drone_aerial_segmentation\archive\semantic_drone_dataset\semantic_drone_dataset' \
                 r'\original_images'
    mask_data_path = r'C:\data\drone_aerial_segmentation\archive\semantic_drone_dataset\semantic_drone_dataset' \
                     r'\label_images_semantic'
    n_classes = 23
    df = create_df(im_data_path)
    print('Total Images: ', len(df))

    # split data
    X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19)
    X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)

    print('Train Size   : ', len(X_train))
    print('Val Size     : ', len(X_val))
    print('Test Size    : ', len(X_test))

    # setup training
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    t_train = TransformerConfig.get_train_transforms()

    t_val = TransformerConfig.get_val_transform()

    # datasets
    train_set = DroneDataset(im_data_path, mask_data_path, X_train, mean, std, t_train, patch=False)
    val_set = DroneDataset(im_data_path, mask_data_path, X_val, mean, std, t_val, patch=False)

    # dataloader
    batch_size = 1

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=23, activation=None, encoder_depth=5,
                     decoder_channels=[256, 128, 64, 32, 16])

    max_lr = 1e-3
    epoch = 15
    weight_decay = 1e-4

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch, steps_per_epoch=len(train_loader))

    trainer = DroneTrainer(device=device, optimizer=optimizer, scheduler=scheduler, criterion=criterion)
    trainer.train_model(epoch, model, train_loader, val_loader)
    exit(0)
    history = fit(epoch, model, train_loader, val_loader, criterion, optimizer, scheduler, device=device)


if __name__ == '__main__':
    main()
