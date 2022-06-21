import torch
from torch.utils.data.dataset import random_split
from lib.mlggm.datasets import Apnea
from torch.utils.data import DataLoader

import pytorch_lightning as pl

class ApneaDataModule(pl.LightningDataModule):

    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            shuffle: bool,
            path: str
    ):
        super(ApneaDataModule, self).__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.persistent_workers = True if self.num_workers > 0 else False

        # load train dataset and split into training and validation sets
        train_ds = Apnea(
            path=path,
            ds_type="train"
        )

        print("Size of training data before split: ", train_ds.X.shape, train_ds.y.shape)

        val_len = round(0.1 * len(train_ds))
        train_len = len(train_ds) - val_len

        self.trainset, self.validset = random_split(
            train_ds, lengths=[train_len, val_len], generator=torch.Generator().manual_seed(42)
        )

        print("Size of training set: ", len(self.trainset))
        print("Size of validation set: ", len(self.validset))

        self.testset = Apnea(
            path=path,
            ds_type="test"
        )
        print("Size of testing set: ", self.testset.X.shape, self.testset.y.shape)

    def train_dataloader(self) -> DataLoader:

        loader = DataLoader(
            self.trainset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True
        )
        return loader

    def val_dataloader(self) -> DataLoader:

        loader = DataLoader(
            self.validset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self) -> DataLoader:

        loader = DataLoader(
            self.testset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True
        )
        return loader
