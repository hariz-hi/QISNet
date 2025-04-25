import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader import BitplaneImageDataset
from dataloaders.patch_dataset_wrapper import PatchBitplaneDatasetWrapper


def get_train_val_dataloaders(
        batch_size, T, train_rate=0.8, num_workers=4, shuffle=True, drop_last=True,
        is_patch=False, patch_size=8, stride=4
):
    assert 0 < train_rate <= 1

    train_val_dataset = BitplaneImageDataset(
        root_dir="dataset/train",
        num_bitplane=T,
    )

    if is_patch:
        train_val_dataset = PatchBitplaneDatasetWrapper(
            train_val_dataset, patch_size=patch_size, stride=stride
        )

    train_len = int(len(train_val_dataset) * train_rate)
    val_len = len(train_val_dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return train_loader, val_loader


def get_test_dataloaders(T, batch_size=1):
    train = BitplaneImageDataset(
        root_dir="dataset/test",
        num_bitplane=T,
    )

    test_loader = DataLoader(
        train,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
    )

    return test_loader
