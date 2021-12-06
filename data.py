from typing import List
import numpy as np
import matplotlib.pyplot as plt
import torch
import xarray as xr
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from matplotlib import cm


class ERA5Data(Dataset):

    def __init__(self, datasets: List[xr.Dataset], patch_size: int, pool_size: int, pool_type: str):
        # each element of data is an xr.Dataset representing a different physical variable
        # in our case, data = [t2m, tp] = [temp @ 2 meters, total precipitation]
        # we can think of each element of data as representing a different image channel
        # we merge these channels into a single c x h x w tensor in __getitem__
        self.datasets = datasets
        self.patch_size = patch_size
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.n_channels = len(self.datasets)
        self.n_months = datasets[0].shape[0]
        self.n_vertical = datasets[0].shape[1] // patch_size
        self.n_horizontal = datasets[0].shape[2] // patch_size
        for dataset in self.datasets:
            mu = np.nanmean(dataset.values)
            s = np.nanstd(dataset.values)
            dataset.values = (dataset.values - mu) / s


    def __len__(self):
        return self.n_months * self.n_vertical * self.n_horizontal

    def __getitem__(self, i: int):
        # e.g., if we have 100 patches per month, then i=527 corresponds to 5th month and 527//100 = 5
        month = i // (self.n_vertical * self.n_horizontal)

        # e.g., i=527 corresponds to 27th patch in image and 527 % 100 = 27
        patch = i % (self.n_vertical * self.n_horizontal)

        # e.g., if each image is (h, w) = (5, 20) then patch 27 corresponds to 2nd row and 27 // 20 = 1
        row = patch // self.n_horizontal

        # e.g., if each image is (h, w) = (5, 20) then patch 27 corresponds to 8th col and 27 % 20 = 7
        col = patch % self.n_horizontal

        # extract patch for this month, this vertical offset, and this horizontal offset by collating channels
        ps = self.patch_size
        input_channels = []
        target_channels = []
        for c in range(self.n_channels):
            # extract one channel at a time from self.datasets list of xr.Datasets
            channel = self.datasets[c][month, row*ps:(row+1)*ps, col*ps:(col+1)*ps].values

            # if more than half of values are nan, skip this region and return None
            # None will be handled properly by collate_fn of DataLoader
            if np.sum(np.isnan(channel)) > ps ** 2 / 2:
                return None
            # otherwise replace remaining nans with mean of region
            else:
                replacement = np.nanmean(channel)
                channel = np.nan_to_num(channel, nan=replacement)

            # this original full-resolution channel is the target
            target_channels.append(channel)

            # downsample and upsample to produce low-resolution input channel
            # https://stackoverflow.com/a/42463514
            if self.pool_type == "mean":
                downsampled = channel.reshape(ps // self.pool_size, self.pool_size,
                                              ps // self.pool_size, self.pool_size).mean(axis=(1, 3))
            elif self.pool_type == "max":
                downsampled = channel.reshape(ps // self.pool_size, self.pool_size,
                                              ps // self.pool_size, self.pool_size).max(axis=(1, 3))
            else:
                raise ValueError("Invalid pooling type.")
            upsampled = np.repeat(np.repeat(downsampled, self.pool_size, axis=0), self.pool_size, axis=1)
            input_channels.append(upsampled)

        # return input x and output y for batch collation
        input = torch.from_numpy(np.array(input_channels))
        target = torch.from_numpy(np.array(target_channels))

        return {"x": input, "y": target}

    def _get_batch(self, years_per_batch):
        # deprecated: gets whole-geographic batch for a given number of years
        inputs, targets = [], []
        for month in range(self.n_months):
            for row in range(self.n_vertical):
                for col in range(self.n_horizontal):
                    ps = self.patch_size
                    input_channels = []
                    target_channels = []
                    for c in range(self.n_channels):
                        # extract one channel at a time from self.datasets list of xr.Datasets
                        channel = self.datasets[c][month, row*ps:(row+1)*ps, col*ps:(col+1)*ps].values

                        # if more than half of values are nan, skip this region and return None
                        # None will be handled properly by collate_fn of DataLoader
                        if np.sum(np.isnan(channel)) > ps**2 / 2:
                            continue
                        # otherwise replace remaining nans with mean of region
                        else:
                            replacement = np.nanmean(channel)
                            channel = np.nan_to_num(channel, nan=replacement)

                        # this original full-resolution channel is the target
                        target_channels.append(channel)

                        # downsample and upsample to produce low-resolution input channel
                        # https://stackoverflow.com/a/42463514
                        if self.pool_type == "mean":
                            downsampled = channel.reshape(ps // self.pool_size, self.pool_size,
                                                          ps // self.pool_size, self.pool_size).mean(axis=(1, 3))
                        elif self.pool_type == "max":
                            downsampled = channel.reshape(ps // self.pool_size, self.pool_size,
                                                          ps // self.pool_size, self.pool_size).max(axis=(1, 3))
                        else:
                            raise ValueError("Invalid pooling type.")
                        upsampled = np.repeat(np.repeat(downsampled, self.pool_size, axis=0), self.pool_size, axis=1)
                        input_channels.append(upsampled)

                    # save input x and output y for batch if valid
                    if len(input_channels) > 0:
                        inputs.append(np.array(input_channels))
                        targets.append(np.array(target_channels))

            if (month // 12) % years_per_batch == 0:
                # return a batch with a few years
                input_tensor = torch.from_numpy(np.array(inputs))
                target_tensor = torch.from_numpy(np.array(targets))
                yield {"x": input_tensor, "y": target_tensor}
                inputs, targets = [], []


class ERA5DataModule(pl.LightningDataModule):

    def __init__(self, args):
        # setup construction parameters
        self.patch_size = args.get("patch_size", 64)
        self.pool_size = args.get("pool_size", 2)
        self.pool_type = args.get("pool_type", "mean")

        # setup data
        self.data = xr.open_dataset(f"era5.nc")
        self.train_start = args.get("train_start", 1950)
        self.train_end = args.get("train_end", 2000)
        self.val_start = args.get("val_start", 2000)
        self.val_end = args.get("val_end", 2010)
        self.test_start = args.get("test_start", 2010)
        self.test_end = args.get("test_end", 2020)
        train_data = [getattr(self.data, x)[12*(self.train_start - 1950):12*(self.train_end - 1950)] for x in ["t2m", "tp"]]
        val_data = [getattr(self.data, x)[12*(self.val_start - 1950):12*(self.val_end - 1950)] for x in ["t2m", "tp"]]
        test_data = [getattr(self.data, x)[12*(self.test_start - 1950):12*(self.test_end - 1950)] for x in ["t2m", "tp"]]
        self.train_data = ERA5Data(train_data, self.patch_size, self.pool_size, self.pool_type)
        self.val_data = ERA5Data(val_data, self.patch_size, self.pool_size, self.pool_type)
        self.test_data = ERA5Data(test_data, self.patch_size, self.pool_size, self.pool_type)

        # setup loader parameters
        self.batch_size = args.get("batch_size", 32)

    def collate_fn(self, batch):
        # get rid of None in minibatch arising from edges of dataset
        # https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806/7
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=self.collate_fn)


if __name__ == "__main__":

    # test entire DataModule with __getitem__ indexing
    datamodule = ERA5DataModule(args={"pool_size": 4, "batch_size": 32})
    dataloader = datamodule.train_dataloader()
    fig, ax = plt.subplots(4, 4, figsize=(12, 12))
    batch = next(iter(dataloader))
    x = batch["x"].detach().cpu().numpy()
    y = batch["y"].detach().cpu().numpy()
    for i in range(4):
        ax[i, 0].imshow(x[i][0], cmap=cm.RdYlBu_r); ax[i, 0].set_title("T2M @ LR")
        ax[i, 1].imshow(y[i][0], cmap=cm.RdYlBu_r); ax[i, 1].set_title("T2M @ HR")
        ax[i, 2].imshow(x[i][1], cmap=cm.BrBG); ax[i, 2].set_title("TP @ LR")
        ax[i, 3].imshow(y[i][1], cmap=cm.BrBG); ax[i, 3].set_title("TP @ HR")
    plt.tight_layout()
    plt.show()

    # test Dataset with deprecated _get_batch indexing
    # dataset = datamodule.train_data
    # fig, ax = plt.subplots(4, 4, figsize=(12, 12))
    # x = next(dataset._get_batch(1))["x"].detach().cpu().numpy()
    # y = next(dataset._get_batch(1))["y"].detach().cpu().numpy()
    # for i in range(4):
    #     ax[i, 0].imshow(x[i][0], cmap=cm.RdYlBu_r); ax[i, 0].set_title("T2M @ LR")
    #     ax[i, 1].imshow(y[i][0], cmap=cm.RdYlBu_r); ax[i, 1].set_title("T2M @ HR")
    #     ax[i, 2].imshow(x[i][1], cmap=cm.BrBG); ax[i, 2].set_title("TP @ LR")
    #     ax[i, 3].imshow(y[i][1], cmap=cm.BrBG); ax[i, 3].set_title("TP @ HR")
    # plt.tight_layout()
    # plt.show()
