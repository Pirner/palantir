import os

import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms as T
import torch
from PIL import Image
import cv2


class DroneWaterSegmentationDataset(Dataset):
    def __init__(
            self,
            im_paths,
            mask_paths,
            mean=0,
            std=1,
            transform=None
    ):
        """
        dataset for water segmentation in flooded areas
        :param im_paths: image paths for segmentation
        :param mask_paths: mask paths for segmentation
        :param mean: mean value for standardization
        :param std: standard deviation
        :param transform: transformation to run
        """
        self.im_paths = im_paths
        self.mask_paths = mask_paths
        self.mean = mean
        self.std = std
        # self.fg_color = fg_color
        self.transform = transform

        assert len(im_paths) == len(mask_paths)

    def __len__(self):
        """
        length of the dataset
        :return:
        """
        return len(self.im_paths)

    def __getitem__(self, idx):
        """
        get an item from the data loading
        :param idx: index to load item from
        :return:
        """
        img = cv2.imread(self.im_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], 0)

        mask_building_flooded = mask == 1
        mask_road_flooded = mask == 3
        mask_water = mask == 5
        mask = mask_road_flooded.astype(int) + mask_building_flooded.astype(int) + mask_water.astype(int)

        mask = mask.astype(float)
        mask = np.expand_dims(mask, axis=-1)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        t = T.Compose([T.ToTensor(), ])
        img = t(img)
        mask = torch.from_numpy(mask).long()
        mask = mask.permute(2, 0, 1)  # channels first
        # return a tuple of the image and its mask
        mask = mask.type(torch.FloatTensor)

        return img, mask
