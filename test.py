import argparse
import pytorch_lightning as pl
from data import ERA5DataModule
from models import SRCNN, VDSR, SRResNet

def main(args):
    e = ERA5DataModule(args={
        "pool_size": args.pool_size,
        "batch_size": 16,
        "patch_size": 64
    })
    test_dl = e.test_dataloader()

    if args.model.lower() == 'srcnn':
        model = SRCNN
    elif args.model.lower() == 'srresnet':
        model = SRResNet
    elif args.model.lower() == 'vdsr':
        model = VDSR

    model = model.load_from_checkpoint(args.checkpoint)

    trainer = pl.Trainer()

    trainer.test(model, test_dl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--model', default="SRCNN", type=str, help="Model to train")
    parser.add_argument('--checkpoint', type=str, help="Checkpoint file (.ckpt)")
    parser.add_argument('--pool_size', default=4, type=int, help="Super-resolution factor")

    args = parser.parse_args()

    main(args)