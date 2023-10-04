import os

from src.training.satellite_image_trainer import SatelliteImageTrainer
from src.constants import forest_backbone


def main():
    fg_color = 0, 255, 0
    dataset_src = r'C:\data\palantir\forest_segmentation'
    trainer = SatelliteImageTrainer(fg_color, batch_size=1, backbone=forest_backbone)

    trainer.train_model(train_path=dataset_src)


if __name__ == '__main__':
    main()
