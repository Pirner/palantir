import torch
import segmentation_models_pytorch as smp

from src.constants import forest_seg_h, forest_seg_w, forest_backbone


def main():
    model_path = r'C:\data\palantir\models\binary_forest\00\model_softmax.pt'
    h, w = forest_seg_h, forest_seg_w
    model = smp.Unet(
        forest_backbone,
        encoder_weights='imagenet',
        classes=1,
        activation=None,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
    )
    device = torch.device('cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    inputs = torch.randn(1, 3, h, w)
    traced_model = torch.jit.trace(model, inputs)
    traced_model.save(r'C:\data\palantir\models\binary_forest\00\model_softmax_traced.pt')


if __name__ == '__main__':
    main()
