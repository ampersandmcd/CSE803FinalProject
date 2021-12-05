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
    """
    Image Super-Resolution Using Deep Convolutional Networks
    Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang
    https://arxiv.org/pdf/1501.00092.pdf
    """
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
            extra_args["padding"] = "same"
            extra_args["padding_mode"] = "replicate"

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_1, kernel_size=self.kernel_1, **extra_args),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden_1, out_channels=self.hidden_2, kernel_size=self.kernel_2, **extra_args),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden_2, out_channels=self.output_dim, kernel_size=self.kernel_3, **extra_args),
        )

    def forward(self, x):
        return self.layers(x)


class VDSR(BaseModel):
    """
    Accurate Image Super-Resolution Using Very Deep Convolutional Networks
    Jiwon Kim, Jung Kwon Lee and Kyoung Mu Lee
    https://arxiv.org/pdf/1511.04587.pdf
    """
    def __init__(
            self,
            d: int = 20,                # d=20 in the paper
            kernel: int = 3,            # k=3 in the paper
            hidden_dim: int = 64,       # hidden_dim=64 in the paper
            **kwargs
    ):
        super().__init__(**kwargs)
        self.d = d
        self.kernel = kernel
        self.hidden_dim = hidden_dim

        layers = []
        layers.append(nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=self.kernel, padding="same", padding_mode="replicate"))
        layers.append(nn.ReLU())
        for _ in range(d-2):
            layers.append(nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=self.kernel, padding="same", padding_mode="replicate"))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.output_dim, kernel_size=self.kernel, padding="same", padding_mode="replicate"))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)


