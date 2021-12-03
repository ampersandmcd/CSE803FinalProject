import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize(data, year, month, variable, save=False):
    if variable == "t2m":
        img = data.t2m[12*(year - 1950) + month - 1]
        name = "Surface Temperature"
        cmap = "RdYlBu_r"
    elif variable == "tp":
        img = data.tp[12*(year - 1950) + month - 1]
        name = "Total Precipitation"
        cmap = "BrBG"
    else:
        raise ValueError("Invalid variable.")
    fig, ax = plt.subplots(figsize=(8, 4))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    divnorm = colors.TwoSlopeNorm(vmin=img.min(), vcenter=img.mean(), vmax=img.max())
    mapping = ax.imshow(img, cmap=cmap, norm=divnorm)
    fig.colorbar(mapping, cax=cax)
    ax.set_title(f"ERA5 {name} ({month}/{year})")
    plt.tight_layout()
    if save:
        plt.savefig(f"figs/{variable}_{year}_{month}.png")
    plt.show()


def visualize_patches(data, year, month, variable, size=32, save=False):
    if variable == "t2m":
        img = data.t2m[12*(year - 1950) + month - 1]
        name = "Surface Temperature"
        cmap = "RdYlBu_r"
    elif variable == "tp":
        img = data.tp[12*(year - 1950) + month - 1]
        name = "Total Precipitation"
        cmap = "BrBG"
    else:
        raise ValueError("Invalid variable.")
    n_vertical, n_horizontal = img.shape[0] // size, img.shape[1] // size
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(n_vertical, n_horizontal + 1, width_ratios=[1 for i in range(n_horizontal)] + [0.1])
    min_, mean_, max_ = img.min(), img.mean(), img.max()
    divnorm = colors.TwoSlopeNorm(vmin=min_, vcenter=mean_, vmax=max_)
    for i in range(n_vertical):
        for j in range(n_horizontal):
            ax = fig.add_subplot(gs[i, j])
            patch = img[i*size:(i+1)*size, j*size:(j+1)*size]
            mapping = ax.imshow(patch, norm=divnorm, cmap=cmap)
            ax.axis("off")
            ax.set_title(f"{i*size}:{(i+1)*size}, {j*size}:{(j+1)*size}", fontsize=6)
    cax = fig.add_subplot(gs[:, -1])
    fig.colorbar(mapping, cax=cax)
    fig.suptitle(f"ERA5 {name} ({month}/{year}) Patches {size}x{size}")
    plt.tight_layout()
    if save:
        plt.savefig(f"figs/{variable}_{year}_{month}_patches.png")
    plt.show()


def visualize_pool(data, year, month, variable, row, col, pool=2, size=32, save=False):
    if variable == "t2m":
        img = data.t2m[12*(year - 1950) + month - 1]
        name = "Surface Temperature"
        cmap = "RdYlBu_r"
    elif variable == "tp":
        img = data.tp[12*(year - 1950) + month - 1]
        name = "Total Precipitation"
        cmap = "BrBG"
    else:
        raise ValueError("Invalid variable.")

    # extract patch and set up figure
    img = img[row:row+size, col:col+size]
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.1])
    min_, mean_, max_ = img.min(), img.mean(), img.max()
    divnorm = colors.TwoSlopeNorm(vmin=min_, vcenter=mean_, vmax=max_)

    # plot original patch
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    mapping = ax.imshow(img, norm=divnorm, cmap=cmap)

    # create downsampled patch of same dimension
    downsampled = img.values.reshape(size//pool, pool, size//pool, pool).mean(axis=(1, 3))
    upsampled = np.repeat(np.repeat(downsampled, pool, axis=0), pool, axis=1)

    # plot downsampled patch
    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    ax.imshow(downsampled, norm=divnorm, cmap=cmap)

    # plot colorbar
    cax = fig.add_subplot(gs[2])
    fig.colorbar(mapping, cax=cax)
    fig.suptitle(f"ERA5 {name} ({month}/{year}) Patch {size}x{size} at ({row},{col}) Pooled {pool}x")
    plt.tight_layout()
    if save:
        plt.savefig(f"figs/{variable}_{year}_{month}_pool_{pool}_{row}_{col}.png")
    plt.show()


if __name__ == "__main__":
    data = xr.open_dataset(f"era5.nc")
    visualize(data, 2020, 4, "t2m")
    visualize(data, 2020, 4, "tp")
    visualize_patches(data, 2020, 4, "t2m", size=32)
    visualize_patches(data, 2020, 4, "tp", size=32)
    visualize_pool(data, 2020, 4, "t2m", row=98, col=192, pool=2)
    visualize_pool(data, 2020, 4, "t2m", row=98, col=192, pool=4)
    visualize_pool(data, 2020, 4, "t2m", row=98, col=192, pool=8)
