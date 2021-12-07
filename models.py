import numpy as np
import pytorch_lightning as pl
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn as nn
import torch.nn.functional as F
import torch


class BaseModel(pl.LightningModule):
    def __init__(
            self,
            input_channels: list = [0, 1],      # indices of tensor input channels to consider (0=t2m, 1=tp)
            output_channels: list = [0, 1],     # indices of tensor target channels to predict (0=t2m, 1=tp)
            lr: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_dim = len(input_channels)
        self.output_dim = len(output_channels)

    def training_step(self, batch, batch_idx):
        x = batch['x'][:, self.input_channels, :, :]
        y = batch['y'][:, self.output_channels, :, :]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x'][:, self.input_channels, :, :]
        y = batch['y'][:, self.output_channels, :, :]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['x'][:, self.input_channels, :, :]
        y = batch['y'][:, self.output_channels, :, :]
        y_hat = self(x)

        def _ssim_trans(x):
            return x.detach().cpu().permute(0, 2, 3, 1).numpy()

        def _psnr_trans(x):
            x_ = x.detach().cpu().numpy()
            min_ = np.amin(x_)
            max_ = np.amax(x_)
            return (x_ - min_) / (max_ - min_)

        self.log_dict({
            'MSE': F.mse_loss(y_hat, y),
            'SSIM': ssim(_ssim_trans(y_hat), _ssim_trans(y), multichannel=True),
            'PSNR': psnr(image_true=_psnr_trans(y), image_test=_psnr_trans(y_hat), data_range=1)
        })

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


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

    def training_step(self, batch, batch_idx):
        x = batch['x'][:, self.input_channels, :, :]
        y = batch['y'][:, self.output_channels, :, :]
        y_hat = self(x)    
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
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
            pre_post_kernel: int = 9,   # not in paper, but we draw inspiration from SRCNN and SRResNet
            **kwargs
    ):
        super().__init__(**kwargs)
        self.d = d
        self.kernel = kernel
        self.hidden_dim = hidden_dim
        self.pre_post_kernel = pre_post_kernel

        # construct layer before blocks
        pre_layers = [
            nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=self.pre_post_kernel, padding="same", padding_mode="replicate"),
            nn.ReLU()
        ]
        self.pre_layers = nn.Sequential(*pre_layers)

        # construct main set of blocks
        blocks = []
        for _ in range(d-2):
            blocks.append(nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=self.kernel, padding="same", padding_mode="replicate"))
            blocks.append(nn.ReLU())
        self.blocks = nn.Sequential(*blocks)

        # construct layer after blocks and residual connection
        post_layers = [
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.output_dim, kernel_size=self.pre_post_kernel, padding="same", padding_mode="replicate"),
        ]
        self.post_layers = nn.Sequential(*post_layers)

    def forward(self, x):
        x = self.pre_layers(x)  # preprocess input to hidden_dim x h x w
        x = x + self.blocks(x)  # apply post-block skip connection
        x = self.post_layers(x)     # postprocess back to c x h x w
        return x


class SRResNet(BaseModel):
    """
    Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, ´
    Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf
    """
    def __init__(
            self,
            n_blocks: int = 16,         # n_blocks=16 in the paper
            kernel: int = 3,            # k=3 in the paper
            pre_post_kernel: int = 9,   # pre_post_kernel=9 in the paper
            hidden_dim: int = 64,       # hidden_dim=64 in the paper
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_blocks = n_blocks
        self.kernel = kernel
        self.pre_post_kernel = pre_post_kernel
        self.hidden_dim = hidden_dim

        # construct layers before residual blocks
        pre_layers = [
            nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=self.pre_post_kernel, padding="same", padding_mode="replicate"),
            nn.PReLU()
        ]
        self.pre_layers = nn.Sequential(*pre_layers)

        # construct residual blocks
        blocks = []
        for _ in range(n_blocks):
            block = [
                nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=self.kernel, padding="same", padding_mode="replicate"),
                nn.BatchNorm2d(num_features=self.hidden_dim),
                nn.PReLU(),
                nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=self.kernel, padding="same", padding_mode="replicate"),
                nn.BatchNorm2d(num_features=self.hidden_dim)
            ]
            block = nn.Sequential(*block)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        # construct layers after residual blocks, before last residual connection
        post_layers_1 = [
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=self.kernel, padding="same", padding_mode="replicate"),
            nn.BatchNorm2d(num_features=self.hidden_dim)
        ]
        self.post_layers_1 = nn.Sequential(*post_layers_1)

        # construct layers after residual blocks, after last residual connection
        post_layers_2 = [
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=self.kernel, padding="same", padding_mode="replicate"),
            nn.PReLU(),
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=self.kernel, padding="same", padding_mode="replicate"),
            nn.PReLU(),
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.output_dim, kernel_size=self.pre_post_kernel, padding="same", padding_mode="replicate"),
        ]
        self.post_layers_2 = nn.Sequential(*post_layers_2)

    def forward(self, x):
        x = self.pre_layers(x)                  # preprocess input to hidden_dim x h x w
        x_pre = x                               # save for residual connections later
        for block in self.blocks:               # apply residual blocks
            x = x + block(x)
        x = x_pre + self.post_layers_1(x)       # apply post-block skip connection
        x = self.post_layers_2(x)               # postprocess back to c x h x w
        return x


class Nearest(BaseModel):
    """
    Baseline: apply nearest-neighbor upscaling from LR to HR. Because our dataloaders automatically apply
    nearest-neighbor upscaling to ensure LR and HR are of same dimension, this model implements the identity function.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        return x


class Bilinear(BaseModel):
    """
    Baseline: apply bilinear interpolation-based upscaling from LR to HR. Because our dataloaders automatically apply
    nearest-neighbor upscaling to ensure LR and HR are of same dimension, we must first downscale then upscale
    to use prebuilt PyTorch code.
    """
    def __init__(self, pool_size, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.pool = nn.AvgPool2d(kernel_size=self.pool_size, stride=self.pool_size)

    def forward(self, x):
        size = x.shape[-1]
        x = self.pool(x)    # note that x is already pooled and repeated; this changes size but does not change content
        x = F.interpolate(x, size=size, mode="bilinear")
        return x


class Bicubic(BaseModel):
    """
    Baseline: apply bicubic interpolation-based upscaling from LR to HR. Because our dataloaders automatically apply
    nearest-neighbor upscaling to ensure LR and HR are of same dimension, we must first downscale then upscale
    to use prebuilt PyTorch code.
    """
    def __init__(self, pool_size, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.pool = nn.AvgPool2d(kernel_size=self.pool_size, stride=self.pool_size)

    def forward(self, x):
        size = x.shape[-1]
        x = self.pool(x)  # note that x is already pooled and repeated; this changes size but does not change content
        x = F.interpolate(x, size=size, mode="bicubic")
        return x
