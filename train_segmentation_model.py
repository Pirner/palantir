import os

from src.training.satellite_image_trainer import SatelliteImageTrainer


def main():
    fg_color = 0, 255, 0
    dataset_src = r'C:\data\palantir\src_data\deep_globe'
    train_path = os.path.join(dataset_src, 'train')
    val_path = os.path.join(dataset_src, 'valid')
    trainer = SatelliteImageTrainer(fg_color, batch_size=2, backbone='timm-efficientnet-b3')

    trainer.train_model(train_path=train_path)


if __name__ == '__main__':
    main()
