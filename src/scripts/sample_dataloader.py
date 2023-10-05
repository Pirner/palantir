from torch.utils.data import DataLoader

from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset


def main():
    root = "."
    # SimpleOxfordPetDataset.download(root)
    # init train, val, test sets
    train_dataset = SimpleOxfordPetDataset(root, "train")
    valid_dataset = SimpleOxfordPetDataset(root, "valid")
    test_dataset = SimpleOxfordPetDataset(root, "test")

    # It is a good practice to check datasets don`t intersects with each other
    assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
    assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    n_cpu = 1
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

    sample = train_dataset[0]
    exit(0)


if __name__ == '__main__':
    main()
