import numpy as np
import pandas as pd


class PalantirMaskUtils:
    """
    palantir mask utils
    """
    @staticmethod
    def convert_mask_from_id_into_id(mask: np.ndarray, n_classes=24) -> np.ndarray:
        """
        convert a mask from rgb into id wise, meaning that
        :param mask: mask to convert
        :param n_classes: number of classes for the onehot encoding
        :return:
        """
        result_mask = np.zeros((mask.shape[0], mask.shape[1], n_classes), dtype=np.uint8)
        for i in range(n_classes):
            id_mask = (mask == i).astype(np.uint8)
            result_mask[:, :, i] = id_mask
        return result_mask
