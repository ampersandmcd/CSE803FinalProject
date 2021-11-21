import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl

class BaseModel(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def training_step(self):
        pass

    def validation_step(self):
        pass

    def configure_optimizers(self):
        pass


class SRCNN(BaseModel):
    def __init__(
        self,
        hidden_1: int = 64, # n_1 in the paper
        hidden_2: int = 32, # n_2 in the paper
        kernel_1: int = 9, # f_1
        kernel_2: int = 1, # f_2
        kernel_3: int = 5, # f_3
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.kernel_3 = kernel_3
        
        self.layers = nn.Sequential(
            nn.Conv2d(self.input_dim, self.hidden_1, kernel_1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(self.hidden_1, self.hidden_2, self.kernel_2, padding='same'),
            nn.ReLU(),
            nn.Conv2d(self.hidden_2, self.output_dim, self.kernel_3, padding='same')
        )

    def training_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        x.unsqueeze_(dim=1)
        y.unsqueeze_(dim=1)
        y_hat = self(x)
        loss = F.mse_loss(y, y_hat)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        x.unsqueeze_(dim=1)
        y.unsqueeze_(dim=1)
        y_hat = self(x)
        loss = F.mse_loss(y, y_hat)
        self.log('val_loss', loss)
        return loss

    def forward(self, x):
        return self.layers(x)