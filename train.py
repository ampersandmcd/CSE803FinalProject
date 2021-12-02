from argparse import ArgumentParser
from models import SRCNN
from data import ERA5Data
import pytorch_lightning as pl


def main(args):
    # Data
    e = ERA5Data(args={"pool_size": 4})
    train_dl, val_dl = e.train_dataloader(), e.val_dataloader()
    
    model = SRCNN()

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)