import argparse
import pytorch_lightning as pl
from data import ERA5DataModule
from models import SRCNN, VDSR, SRResNet

def main(args):
    e = ERA5DataModule(args={
        "pool_size": args.pool_size,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size
    })
    test_dl = e.test_dataloader()

    if args.model.lower() == 'srcnn':
        model = SRCNN
    elif args.model.lower() == 'srresnet':
        model = SRResNet
    elif args.model.lower() == 'vdsr':
        model = VDSR

    model = model.load_from_checkpoint(args.checkpoint)

    # Wandb logging
    wandb_logger = pl.loggers.WandbLogger(project='cv-proj')
    wandb_logger.watch(model, log_freq=500)

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = wandb_logger

    trainer.test(model, test_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--model', default="SRCNN", type=str, help="Model to test")
    parser.add_argument('--checkpoint', type=str, help="Checkpoint file (.ckpt)")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size to train with")
    parser.add_argument('--pool_size', default=4, type=int, help="Super-resolution factor")
    parser.add_argument('--patch_size', default=64, type=int, help="Image patch size to super-resolve")

    args = parser.parse_args()

    main(args)
