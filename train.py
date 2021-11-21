from models import SRCNN
from data import ERA5Data
import pytorch_lightning as pl

if __name__ == "__main__":
    e = ERA5Data(args={"pool_size": 4})
    model = SRCNN()
    train_dl, val_dl = e.train_dataloader(), e.val_dataloader()
    trainer = pl.Trainer()
    trainer.fit(model, train_dl, val_dl)
