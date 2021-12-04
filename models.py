import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl


class BaseModel(pl.LightningModule):
    def __init__(
            self,
            input_channels: list = [0, 1],      # indices of tensor input channels to consider (0=t2m, 1=tp)
            output_channels: list = [0, 1],     # indices of tensor target channels to predict (0=t2m, 1=tp)
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_dim = len(input_channels)
        self.output_dim = len(output_channels)

    def training_step(self, batch, batch_idx):
        x = batch['x'][:, self.input_channels, :, :]
        y = batch['y'][:, self.output_channels, :, :]
        y_hat = self(x)
        loss = F.mse_loss(y, y_hat)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x'][:, self.input_channels, :, :]
        y = batch['y'][:, self.output_channels, :, :]
        y_hat = self(x)
        loss = F.mse_loss(y, y_hat)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


class SRCNN(BaseModel):
    def __init__(
            self,
            hidden_1: int = 64,     # n_1 in the paper
            hidden_2: int = 32,     # n_2 in the paper
            kernel_1: int = 9,      # f_1
            kernel_2: int = 1,      # f_2
            kernel_3: int = 5,      # f_3
            padding: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.kernel_3 = kernel_3
        self.padding = padding

        extra_args = {}
        if self.padding:
            extra_args['padding'] = 'same'
            extra_args['padding_mode'] = 'zeros'

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_1, kernel_size=self.kernel_1, **extra_args),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden_1, out_channels=self.hidden_2, kernel_size=self.kernel_2, **extra_args),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden_2, out_channels=self.output_dim, kernel_size=self.kernel_3, **extra_args),
        )

    def training_step(self, batch, batch_idx):
        x = batch['x'][:, self.input_channels, :, :]
        y = batch['y'][:, self.output_channels, :, :]
        y_hat = self(x)    
        o = 6 # Hardcoded the offset of where the padding effects are present #(y.shape[-2] - y_hat.shape[-2]) // 2
        loss = F.mse_loss( # Look at only the non-padding portions
            y_hat[:,:,o:o + y_hat.shape[-2], o:o + y_hat.shape[-2]],
            y[:,:,o:o + y_hat.shape[-2], o:o + y_hat.shape[-2]])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x'][:, self.input_channels, :, :]
        y = batch['y'][:, self.output_channels, :, :]
        y_hat = self(x)
        if not self.padding:
            o = (y.shape[-2] - y_hat.shape[-2]) // 2
            loss = F.mse_loss(y_hat, y[:,:,o:o + y_hat.shape[-2], o:o + y_hat.shape[-2]])
        else:
            loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def forward(self, x):
        return self.layers(x)
