import os
import glob

from tqdm import tqdm
import numpy as np
from torchvision import transforms as T
import cv2
import torch

from src.utils import move_to
from src.modeling.transformation import TransformerConfig


def main():
    model_path = r'..\..\baseline_traced.pt'
    dataset_path = r'C:\data\palantir\src_data\deep_globe\valid'
    traced_model = torch.jit.load(model_path)
    result_dir = r'C:\data\palantir\results'

    im_paths = glob.glob(os.path.join(dataset_path, '**/*.jpg'), recursive=True)
    val_transforms = TransformerConfig.get_val_transform()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    print('found {0} image for inference'.format(len(im_paths)))

    for i, im_p in tqdm(enumerate(im_paths), total=len(im_paths)):
        img = cv2.imread(im_p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocessed = val_transforms(image=img)['image']

        t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        preprocessed = t(preprocessed)
        preprocessed = torch.unsqueeze(preprocessed, 0)
        inputs = move_to(preprocessed, 'cuda')
        y_hat = traced_model(inputs)
        y_hat_squeezed = torch.squeeze(y_hat)
        prediction = y_hat_squeezed.detach().cpu().numpy()

        pred_mask = (prediction > 0.5).astype(np.uint8) * 255
        pred_mask_resized = cv2.resize(pred_mask, img.shape[:2])
        cv2.imwrite(os.path.join(result_dir, '{:04d}.png'.format(i)), img)
        cv2.imwrite(os.path.join(result_dir, '{:04d}_mask.png'.format(i)), pred_mask_resized)


if __name__ == '__main__':
    main()
