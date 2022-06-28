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

        self.trainset = Apnea(path=path, ds_type="train")
        self.validset = Apnea(path=path, ds_type="test")

        print("Size of training set: ", len(self.trainset))
        print("Size of validation set: ", len(self.validset))

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
