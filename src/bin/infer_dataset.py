import os
import glob

from torchvision import transforms as T
import cv2
import torch

from src.modeling.transformation import TransformerConfig


def main():
    model_path = r'C:\data\palantir\models\binary_forest\00\model_softmax_traced.pt'
    traced_model = torch.jit.load(model_path)

    im_paths = glob.glob(os.path.join(r'C:\data\palantir\src_data\deep_globe\test', '**/*.jpg'), recursive=True)
    val_transforms = TransformerConfig.get_val_transform()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    print('found {0} image for inference'.format(len(im_paths)))

    for im_p in im_paths:
        img = cv2.imread(im_p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocessed = val_transforms(image=img)['image']

        t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        preprocessed = t(preprocessed)
        preprocessed = torch.unsqueeze(preprocessed, 0)
        y_hat = traced_model(preprocessed)
        y_hat_squeezed = torch.squeeze(y_hat)
        prediction = y_hat_squeezed.detach().cpu().numpy()
        exit(0)


if __name__ == '__main__':
    main()
