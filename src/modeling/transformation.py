import cv2
import albumentations as A

from src.constants import forest_seg_h, forest_seg_w


class TransformerConfig:
    @staticmethod
    def get_train_transforms():
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        t_train = A.Compose([
            A.Resize(forest_seg_h, forest_seg_w, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
            A.GaussNoise()]
        )

        return t_train

    @staticmethod
    def get_val_transform():
        t_val = A.Compose([A.Resize(608, 608, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                           A.GridDistortion(p=0.2)])

        return t_val
