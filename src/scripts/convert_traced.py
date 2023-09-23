import torch
import segmentation_models_pytorch as smp


def main():
    model_path = r'C:\projects\palantir\models\model_1.pt'
    h, w = 704, 1056
    model = smp.Unet(
        'mobilenet_v2',
        encoder_weights='imagenet',
        classes=23,
        activation=None,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
    )
    device = torch.device('cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    inputs = torch.randn(1, 3, h, w)
    traced_model = torch.jit.trace(model, inputs)
    traced_model.save(r'C:\projects\palantir\models\traced_model_1.pt')


if __name__ == '__main__':
    main()
