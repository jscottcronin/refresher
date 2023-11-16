import lightning as L
import torch

import torch.nn.functional as F
import torchmetrics as tm
from torch.utils.data import DataLoader, random_split
import torchvision as tv


class MNISTModel(L.LightningModule):
    def __init__(self, lr=0.01, n_features=28 * 28, n_classes=10, dropout=0.2):
        super().__init__()
        self.lr = lr
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout = dropout

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.n_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(128, self.n_classes),
        )

        self.train_accuracy = tm.Accuracy("multiclass", num_classes=self.n_classes)
        self.val_accuracy = tm.Accuracy("multiclass", num_classes=self.n_classes)
        self.test_accuracy = tm.Accuracy("multiclass", num_classes=self.n_classes)
        self.save_hyperparameters()

    def forward(self, x):
        logits = torch.nn.functional.log_softmax(self.model(x), dim=1)
        return logits

    def _shared(self, batch):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.nll_loss(logits, y)
        return x, y, logits, loss

    def training_step(self, batch, batch_idx):
        _, _, _, loss = self._shared(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, y, logits, loss = self._shared(batch)
        y_pred = torch.argmax(logits, dim=1)
        self.val_accuracy.update(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, y, logits, loss = self._shared(batch)
        y_pred = torch.argmax(logits, dim=1)
        self.test_accuracy.update(y_pred, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./data", bs=64):
        super().__init__()
        self.data_dir = data_dir
        self.bs = bs
        self.dl_params = dict(
            batch_size=self.bs, num_workers=8, persistent_workers=True
        )

    def prepare_data(self):
        # download
        tv.datasets.MNIST(self.data_dir, train=True, download=True)
        tv.datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        tf = tv.transforms.Compose(
            [tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.test = tv.datasets.MNIST(self.data_dir, train=False, transform=tf)
        self.predict = tv.datasets.MNIST(self.data_dir, train=False, transform=tf)
        train = tv.datasets.MNIST(self.data_dir, train=True, transform=tf)
        self.train, self.val = torch.utils.data.random_split(train, [55000, 5000])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, **self.dl_params)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, **self.dl_params)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, **self.dl_params)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict, **self.dl_params)


if __name__ == "__main__":
    dm = MNISTDataModule(bs=512)
    m = MNISTModel(lr=1e-3, n_features=28 * 28, n_classes=10, dropout=0.3)
    logger = L.pytorch.loggers.TensorBoardLogger(".")
    trainer = L.Trainer(max_epochs=16, logger=logger)
    trainer.fit(m, datamodule=dm)

    val_acc = trainer.validate(datamodule=dm)[0]["val_acc"]
    test_acc = trainer.test(datamodule=dm)[0]["test_acc"]
    print(f" | Val Acc {val_acc*100:.2f}%" f" | Test Acc {test_acc*100:.2f}%")
