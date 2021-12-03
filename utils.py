import torch
import wandb
import pytorch_lightning as pl


class ImageVisCallback(pl.Callback):
    def __init__(self, val_samples, max_samples=2):
        super().__init__()

        self.val_imgs = val_samples[0]['x']

    def on_validation_end(self, trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
        imgs = self.val_imgs.to(device=model.device).unsqueeze(0)
        imgs = imgs[:, model.output_channels, :, :]

        upresed = model(imgs)

        mosaics = torch.cat([imgs, upresed], dim=-2)
        caption = "Top: Low Res, Bottom: High Res"
        trainer.logger.experiment.log({
            "val/examples": [wandb.Image(mosaic, caption) for mosaic in mosaics],
            "global_step": trainer.global_step # This will make sure wandb gets the epoch/step right
        })