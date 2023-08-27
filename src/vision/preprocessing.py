import cv2
import numpy as np


class PalantirPreprocessor:
    def __init__(self, im_h: int, im_w: int):
        """
        preprocessor module for palantir application
        :param im_h: height of the image for the model
        :param im_w: width of the image for the model
        """
        self.h = im_h
        self.w = im_w

    def preprocess_im_mask_bundle(self, im: np.ndarray, mask: np.ndarray):
        resized = cv2.resize(im, (self.w, self.h))
        resized_mask = cv2.resize(mask, (self.w, self.h), cv2.INTER_NEAREST_EXACT)
        return resized, resized_mask
