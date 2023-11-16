from ISLP import load_data
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import loggers as pl_loggers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics as tm


class SimpleFeedFoward(L.LightningModule):
    def __init__(self, input_size, lr=1e-1, dropout=0.3):
        super().__init__()

        self.lr = lr
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, input_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(input_size * 2),
            nn.Linear(input_size * 2, input_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size, 1),
        )

    def forward(self, x):
        return self.model(x).squeeze()

    def _shared_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.l1_loss(y_pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class BaseballDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.dl_params = dict(
            batch_size=self.batch_size, num_workers=8, persistent_workers=True
        )

    def setup(self, stage=None):
        X, y = load_baseball_data()
        ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
        splits = round(0.8 * len(ds)), round(0.2 * len(ds))
        self.train, self.test = torch.utils.data.random_split(ds, [*splits])
        self.val = self.test
        self.dims = self.train[0][0].shape

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, **self.dl_params)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, **self.dl_params)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, **self.dl_params)


def load_baseball_data():
    df = load_data("Hitters")
    cols_to_drop = ["League", "Division", "NewLeague"]
    df = df.drop(cols_to_drop, axis=1).dropna()
    x = df.drop("Salary", axis=1).values.astype(np.float32)
    y = df["Salary"].values.astype(np.float32)
    return x, y


if __name__ == "__main__":
    data = BaseballDataModule(batch_size=256)
    data.setup()
    model = SimpleFeedFoward(data.dims[0], lr=5e-2, dropout=0.3)
    logger = L.pytorch.loggers.TensorBoardLogger(save_dir="")
    trainer = L.Trainer(max_epochs=400, logger=logger)
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)
