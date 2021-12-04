from argparse import ArgumentParser
from models import SRCNN
from data import ERA5DataModule
import pytorch_lightning as pl
import wandb

from utils import ImageVisCallback

wandb.init(project='cv-proj', entity="cv803f21-superres")


def main(args):
    # Data
    e = ERA5DataModule(args={"pool_size": 4, "batch_size": 64, "patch_size": 64})
    train_dl, val_dl = e.train_dataloader(), e.val_dataloader()
    val_samples = [e.val_data[10]]

    # input channels controls which channels we use as predictors
    # output channels controls which channels we use as targets, i.e., loss signal
    # channel 0 corresponds to t2m and channel 1 corresponds to tp
    # e.g., input_channels=[0, 1], output_channels=[1] predicts tp @ HR using t2m AND tp @ LR
    # e.g., input_channels=[1],    output_channels=[1] predicts tp @ HR using ONLY tp @ LR
    # ...etc.
    model = SRCNN(input_channels=[0,1], output_channels=[0,1])

    # Wandb logging
    wandb_logger = pl.loggers.WandbLogger(project='cv-proj')
    wandb_logger.watch(model, log_freq=500)

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = wandb_logger
    trainer.callbacks.append(ImageVisCallback(val_samples))

    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
