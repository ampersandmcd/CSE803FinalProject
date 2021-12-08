import argparse
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib import colors
from data import ERA5DataModule
from models import SRCNN, VDSR, SRResNet, Nearest, Bilinear, Bicubic


def visualize_preds(dl, models, suptitle="Predictions", file_name="preds", save=True):

    batch = list(dl.dataset._get_batch(years_per_batch=1, start_month=108))[3]
    x = batch["x"].detach().cpu().numpy() * dl.dataset.stds + dl.dataset.mus
    y = batch["y"].detach().cpu().numpy() * dl.dataset.stds + dl.dataset.mus
    y_hats = {}
    for model_name, model in models.items():
        y_hats[model_name] = model(batch["x"]).detach().cpu().numpy() * dl.dataset.stds + dl.dataset.mus

    # extract patch indices we care about
    patches = {
        "WA": 0,
        "MI": 6,
        "UT": 10,
        "PA": 15,
        "TX": 25
    }

    fig = plt.figure(figsize=(20, 2*len(y_hats) + 4))
    gs = fig.add_gridspec(len(y_hats) + 2, 12, width_ratios=[1]*10 + [0.1]*2)
    
    t2m_mins, t2m_means, t2m_maxs = [], [], []
    tp_mins, tp_means, tp_maxs = [], [], []
    for z in [x, y] + list(y_hats.values()):
        t2m_mins.append(z[:, 0, :, :].min())
        t2m_means.append(z[:, 0, :, :].mean())
        t2m_maxs.append(z[:, 0, :, :].max())
        tp_mins.append(z[:, 1, :, :].min())
        tp_means.append(z[:, 1, :, :].mean())
        tp_maxs.append(z[:, 1, :, :].max())
    divnorm_t2m = colors.TwoSlopeNorm(vmin=min(t2m_mins), vcenter=np.mean(t2m_means), vmax=max(t2m_maxs))
    divnorm_tp = colors.TwoSlopeNorm(vmin=min(tp_mins), vcenter=np.mean(tp_means), vmax=max(tp_maxs))

    # plot images
    all_axes = []
    for i, (state, patch) in enumerate(patches.items()):
        x_t2m, y_t2m = x[patch, 0], y[patch, 0]
        y_hats_t2m = [y_hat[patch, 0] for y_hat in y_hats.values()]
        x_tp, y_tp = x[patch, 1], y[patch, 1]
        y_hats_tp = [y_hat[patch, 1] for y_hat in y_hats.values()]

        # t2m
        imgs = [x_t2m] + y_hats_t2m + [y_t2m]
        axes = []
        for j in range(len(imgs)):
            ax = fig.add_subplot(gs[j, 2*i])
            ax.set_xticks([])
            ax.set_yticks([])
            mapping_t2m = ax.imshow(imgs[j], norm=divnorm_t2m, cmap="RdYlBu_r")
            axes.append(ax)
        all_axes.append(axes)

        # tp
        imgs = [x_tp] + y_hats_tp + [y_tp]
        axes = []
        for j in range(len(imgs)):
            ax = fig.add_subplot(gs[j, 2*i + 1])
            ax.set_xticks([])
            ax.set_yticks([])
            mapping_tp = ax.imshow(imgs[j], norm=divnorm_tp, cmap="BrBG")
            axes.append(ax)
        all_axes.append(axes)

    # set up titles
    all_axes = np.array(all_axes).T
    all_axes[0, 0].set_ylabel("LR Input")
    for i, model_name in enumerate(models.keys()):
        all_axes[i+1, 0].set_ylabel(model_name)
    all_axes[-1, 0].set_ylabel("HR Truth")

    variables = ["T2M", "TP"]
    for i, (state, patch) in enumerate(patches.items()):
        all_axes[0, 2*i].set_title(f"{state} ({variables[0]})")
        all_axes[0, 2*i+1].set_title(f"{state} ({variables[1]})")

    # set up colorbars
    t2m_cax = fig.add_subplot(gs[:, -2])
    tp_cax = fig.add_subplot(gs[:, -1])
    fig.colorbar(mapping_t2m, cax=t2m_cax)
    fig.colorbar(mapping_tp, cax=tp_cax)

    plt.suptitle(suptitle)
    plt.tight_layout()
    if save:
        plt.savefig(f"figs/{file_name}.png")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size to train with")
    parser.add_argument('--pool_size', default=4, type=int, help="Super-resolution factor")
    parser.add_argument('--patch_size', default=64, type=int, help="Image patch size to super-resolve")
    args = parser.parse_args()
    args.gpus = 1
    models = {
        "Nearest": Nearest(),
        "Bilinear": Bilinear(pool_size=4),
        "Bicubic": Bicubic(pool_size=4),
        "SRCNN": SRCNN.load_from_checkpoint("cv-proj/SRCNN-lr4-lyric-tree-103-epoch=433-step=73345.ckpt"),
        "VDSR": VDSR.load_from_checkpoint("cv-proj/VDSR-lr4-balmy-sound-102-epoch=86-step=14702.ckpt"),
        "SRResNet": SRResNet.load_from_checkpoint("cv-proj/SRResNet-lr4-robust-capybara-101-epoch=45-step=15547.ckpt")
    }
    e = ERA5DataModule(args={
        "pool_size": args.pool_size,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size
    })
    test_dl = e.test_dataloader()

    # test on usual 4x for April 2020
    visualize_preds(test_dl, models, suptitle=f"Predictions 4x (4/2020)", file_name=f"preds_2020_4_pool_4.png")

    # test also on 8x for April 2020
    eval_dl_8x = e.eval_dataloader(pool_size=8)
    models["Bilinear"] = Bilinear(pool_size=8)
    models["Bicubic"] = Bilinear(pool_size=8)
    visualize_preds(eval_dl_8x, models, suptitle=f"Predictions 8x (4/2020)", file_name=f"preds_2020_4_pool_8.png")

    # test also on 16x for April 2020
    eval_dl_16x = e.eval_dataloader(pool_size=16)
    models["Bilinear"] = Bilinear(pool_size=16)
    models["Bicubic"] = Bilinear(pool_size=16)
    visualize_preds(eval_dl_16x, models, suptitle=f"Predictions 16x (4/2020)", file_name=f"preds_2020_4_pool_16.png")
