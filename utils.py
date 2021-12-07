import os
import torch
import wandb
import pytorch_lightning as pl


class ImageVisCallback(pl.Callback):
    def __init__(self, val_Dataloader, max_samples=10):
        super().__init__()

        self.valLoader = val_Dataloader
        self.max_samples = max_samples

    def on_validation_end(self, trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
        '''imgsA = self.val_imgs.to(device=model.device).unsqueeze(0)
        imgsA = imgsA[:, model.output_channels, :, :]

        imgsY = self.val_y.to(device=model.device).unsqueeze(0)
        imgsY = imgsY[:, model.output_channels, :, :]
        '''

        val_dl = self.valLoader
        dataiter = iter(val_dl)
        for i in range(self.max_samples):
            test = dataiter.next()

            imgs = test['x'][0].to(device=model.device).unsqueeze(0)
            imgs = imgs[:, model.output_channels, :, :]

            imgsY = test['y'][0].to(device=model.device).unsqueeze(0)
            imgsY = imgsY[:, model.output_channels, :, :]

            upresed = model(imgs)

            mosaics = torch.cat([imgs, upresed, imgsY], dim=-2)
            caption = "Image {}: Top: Low Res, Middle: High Res Prediction, Bottom: High Res Truth".format(i)

            logname = "val/examples{}".format(i) if os.name != "nt" else "val\examples{}".format(i)
            trainer.logger.experiment.log({
                logname: [wandb.Image(mosaic, caption) for mosaic in mosaics],
            })

        trainer.logger.experiment.log({
            "global_step": trainer.global_step  # This will make sure wandb gets the epoch/step right
        })
