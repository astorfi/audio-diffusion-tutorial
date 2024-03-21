from math import pi
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from einops import rearrange, repeat
from .utils import default

class Distribution:
    def call(self, num_samples: int, device: torch.device) -> Tensor:
        raise NotImplementedError()

class UniformDistribution(Distribution):
    def init(self, vmin: float = 0.0, vmax: float = 1.0):
        super().init()
        self.vmin, self.vmax = vmin, vmax

    def call(self, num_samples: int, device: torch.device = torch.device(‘cpu’)) -> Tensor:
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin

def pad_dims(x: Tensor, ndim: int) -> Tensor:
    return x.view(*x.shape, *((1,) * ndim))

def clip(x: Tensor, dynamic_threshold: float = 0.0) -> Tensor:
    if dynamic_threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    else:
        x_flat = rearrange(x, ‘b … -> b (…)’)  # Dynamic thresholding
        scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1).clamp(min=1.0)
        scale = pad_dims(scale, ndim=x.ndim - scale.ndim)
        return x.clamp(-scale, scale) / scale

def extend_dim(x: Tensor, dim: int) -> Tensor:
    return x.view(*x.shape + (1,) * (dim - x.ndim))

class Diffusion(nn.Module):
    pass

class VDiffusion(Diffusion):
    def init(self, net: nn.Module, sigma_distribution: Distribution = UniformDistribution(), loss_fn: Any = F.mse_loss):
        super().init()
        self.net = net
        self.sigma_distribution = sigma_distribution
        self.loss_fn = loss_fn

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        batch_size, device = x.shape[0], x.device
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        noise = torch.randn_like(x)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        v_pred = self.net(x_noisy, sigmas, **kwargs)
        return self.loss_fn(v_pred, v_target)

class ARVDiffusion(Diffusion):
    def init(self, net: nn.Module, length: int, num_splits: int, loss_fn: Any = F.mse_loss):
        super().init()
        assert length % num_splits == 0, “length must be divisible by num_splits”
        self.net = net
        self.length = length
        self.num_splits = num_splits
        self.split_length = length // num_splits
        self.loss_fn = loss_fn

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        b, _, t, device, dtype = *x.shape, x.device, x.dtype
        assert t == self.length, “input length must match length”
        sigmas = torch.rand((b, 1, self.num_splits), device=device, dtype=dtype)
        sigmas = repeat(sigmas, “b 1 n -> b 1 (n l)”, l=self.split_length)
        noise = torch.randn_like(x)
        alphas, betas = self.get_alpha_beta(sigmas)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        channels = torch.cat([x_noisy, sigmas], dim=1)
        v_pred = self.net(channels, **kwargs)
        return self.loss_fn(v_pred, v_target)
