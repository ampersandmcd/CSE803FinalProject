import numpy as np
import matplotlib.pyplot as plt
import torch
import xarray as xr
import pytorch_lightning as pl


class ERA5Data(pl.LightningDataModule):

    def __init__(self, args):

        # load and extract data
        self.data = xr.open_dataset(f"era5.nc")
        self.train_start = args.get("train_start", 1950)
        self.train_end = args.get("train_end", 2000)
        self.val_start = args.get("val_start", 2000)
        self.val_end = args.get("val_end", 2010)
        self.test_start = args.get("test_start", 2010)
        self.test_end = args.get("test_end", 2020)
        self.variable = args.get("variable", "t2m")
        self.train_data = getattr(self.data, self.variable)[12*(self.train_start - 1950):12*(self.train_end - 1950)]
        self.val_data = getattr(self.data, self.variable)[12*(self.val_start - 1950):12*(self.val_end - 1950)]
        self.test_data = getattr(self.data, self.variable)[12*(self.test_start - 1950):12*(self.test_end - 1950)]

        # load and extract loader parameters
        self.patch_size = args.get("patch_size", 32)
        self.years_per_batch = args.get("years_per_batch", 5)
        self.pool_size = args.get("pool_size", 2)
        self.pool_type = args.get("pool_type", "mean")

    def get_batch(self, data):
        ps = self.patch_size
        n_months, n_vertical, n_horizontal = data.shape[0], data.shape[1] // ps, data.shape[2] // ps
        inputs, targets = [], []
        for month in range(n_months):
            for i in range(n_vertical):
                for j in range(n_horizontal):
                    # extract patch for this month, this vertical offset, and this horizontal offset
                    patch = data[month, i*ps:(i+1)*ps, j*ps:(j+1)*ps].values

                    # if more than half of values are nan, skip this patch
                    if np.sum(np.isnan(patch)) > ps**2 / 2:
                        continue

                    # replace remaining nans with mean of patch
                    replacement = np.nanmean(patch)
                    patch = np.nan_to_num(patch, nan=replacement)

                    # downsample and upsample to produce input x
                    if self.pool_type == "mean":
                        downsampled = patch.reshape(ps//self.pool_size, self.pool_size, ps//self.pool_size, self.pool_size).mean(axis=(1, 3))
                    elif self.pool_type == "max":
                        downsampled = patch.reshape(ps//self.pool_size, self.pool_size, ps//self.pool_size, self.pool_size).max(axis=(1, 3))
                    else:
                        raise ValueError("Invalid pooling type.")
                    upsampled = np.repeat(np.repeat(downsampled, self.pool_size, axis=0), self.pool_size, axis=1)

                    # save input x and output y for batch
                    inputs.append(upsampled)
                    targets.append(patch)

            if (month // 12) % self.years_per_batch == 0:
                # return a batch with a few years
                input_tensor = torch.from_numpy(np.array(inputs))
                target_tensor = torch.from_numpy(np.array(targets))
                yield {"x": input_tensor, "y": target_tensor}
                inputs, targets = [], []

    def train_dataloader(self):
        return self.get_batch(self.train_data)

    def val_dataloader(self):
        return self.get_batch(self.val_data)

    def test_dataloader(self):
        return self.get_batch(self.test_data)

if __name__ == "__main__":
    e = ERA5Data(args={"pool_size": 4})
    td = e.train_dataloader()
    batch = next(td)
    fig, ax = plt.subplots(6, 2, figsize=(4, 12))
    for i in range(6):
        ax[i, 0].imshow(batch["x"][i])
        ax[i, 1].imshow(batch["y"][i])
    plt.show()