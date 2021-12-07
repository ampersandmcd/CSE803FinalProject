import argparse
import pytorch_lightning as pl
from data import ERA5DataModule
from models import SRCNN, VDSR, SRResNet, Nearest, Bilinear, Bicubic

def main(args):
    e = ERA5DataModule(args={
        "pool_size": args.pool_size,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size
    })
    test_dl = e.test_dataloader()

    if args.model.lower() == 'srcnn':
        print("Testing SRCNN")
        model = SRCNN
        model = model.load_from_checkpoint(args.checkpoint)
    elif args.model.lower() == 'srresnet':
        print("Testing SRResNet")
        model = SRResNet
        model = model.load_from_checkpoint(args.checkpoint)
    elif args.model.lower() == 'vdsr':
        print("Testing VDSR")
        model = VDSR
        model = model.load_from_checkpoint(args.checkpoint)
    elif args.model.lower() == 'nearest':
        print("Testing Nearest")
        model = Nearest()
    elif args.model.lower() == 'bilinear':
        print("Testing Bilinear")
        model = Bilinear(pool_size=args.pool_size)
    elif args.model.lower() == 'bicubic':
        print("Testing Bicubic")
        model = Bicubic(pool_size=args.pool_size)

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
    args.gpus = 1
    # args.model = "SRCNN"
    # args.checkpoint = "cv-proj/SRCNN-lr4-lyric-tree-103-epoch=433-step=73345.ckpt"
    # args.model = "VDSR"
    # args.checkpoint = "cv-proj/VDSR-lr4-balmy-sound-102-epoch=86-step=14702.ckpt"
    # args.model = "SRResNet"
    # args.checkpoint = "cv-proj/SRResNet-lr4-robust-capybara-101-epoch=45-step=15547.ckpt"

    print(f"Loading checkpoint {args.checkpoint}")
    main(args)
