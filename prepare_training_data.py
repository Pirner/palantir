import os
import glob

from tqdm import tqdm
import cv2

from src.vision.preprocessing import PalantirPreprocessor


def main():
    im_w = 640
    im_h = 448

    preprocessor = PalantirPreprocessor(im_h=im_h, im_w=im_w)
    dst_dir = r'C:\data\drone_aerial_segmentation\archive\preprocessed_data'
    dataset_path = r'C:\data\drone_aerial_segmentation\archive\semantic_drone_dataset\semantic_drone_dataset'
    mask_paths = glob.glob(os.path.join(dataset_path, 'label_images_semantic', '**/*.png'), recursive=True)
    im_paths = glob.glob(os.path.join(dataset_path, 'original_images', '**/*.jpg'), recursive=True)

    i = 0
    for im_p, mask_p in tqdm(zip(im_paths, mask_paths), total=len(im_paths)):
        im = cv2.imread(im_p)
        mask = cv2.imread(mask_p, 0)
        im_pre, mask_pre = preprocessor.preprocess_im_mask_bundle(im, mask)
        cv2.imwrite(os.path.join(dst_dir, '{:06d}_im.png'.format(i)), im_pre)
        cv2.imwrite(os.path.join(dst_dir, '{:06d}_mask.png'.format(i)), mask_pre)
        i += 1


if __name__ == '__main__':
    main()
