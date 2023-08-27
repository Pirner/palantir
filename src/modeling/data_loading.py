import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from src.vision.mask_utils import PalantirMaskUtils


class AerialSegmentationSemanticDataset(Dataset):
    def __init__(self, im_paths, mask_paths, augmentation=None, preprocessing=None):
        """

        :param im_paths:
        :param mask_paths:
        :param augmentation:
        :param preprocessing:
        """
        self.im_paths = im_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, i):
        image = cv2.imread(self.im_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[i], 0)

        # print(self.im_paths[i])

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        mask = PalantirMaskUtils.convert_mask_from_id_into_id(mask.detach().cpu().numpy())
        mask = torch.from_numpy(mask.astype(float))
        mask = mask.permute(2, 0, 1)

        return image, mask
