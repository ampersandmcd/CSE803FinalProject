import os
from argparse import ArgumentParser
from models import SRCNN, VDSR, SRResNet
from data import ERA5DataModule
import pytorch_lightning as pl
import wandb

from utils import ImageVisCallback

wandb.init(project='cv-proj', entity="cv803f21-superres")


def main(args):
    # Data
    e = ERA5DataModule(args={
        "pool_size": args.pool_size if hasattr(args, "pool_size") else 4,
        "batch_size": args.batch_size if hasattr(args, "batch_size") else 128,
        "patch_size": args.patch_size if hasattr(args, "patch_size") else 64
    })
    train_dl, val_dl = e.train_dataloader(), e.val_dataloader()
    val_samples = [e.val_data[10]]

    # input channels controls which channels we use as predictors
    # output channels controls which channels we use as targets, i.e., loss signal
    # channel 0 corresponds to t2m and channel 1 corresponds to tp
    # e.g., input_channels=[0, 1], output_channels=[1] predicts tp @ HR using t2m AND tp @ LR
    # e.g., input_channels=[1],    output_channels=[1] predicts tp @ HR using ONLY tp @ LR
    # ...etc.
    args.model = args.model if hasattr(args, "model") else "SRCNN"
    if args.model == "VDSR":
        print("Constructing VDSR")
        model = VDSR(input_channels=[0, 1], output_channels=[0, 1])
    elif args.model == "SRResNet":
        print("Constructing SRResNet")
        model = SRResNet(input_channels=[0, 1], output_channels=[0, 1])
    else:
        print("Constructing SRCNN")
        model = SRCNN(input_channels=[0, 1], output_channels=[0, 1])

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
    parser.add_argument('--model')
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()

    main(args)
