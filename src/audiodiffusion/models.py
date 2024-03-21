from abc import ABC, abstractmethod
from math import floor
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from einops import pack, unpack, rearrange
from torch import Generator, Tensor, nn

from .components import AppendChannelsPlugin, MelSpectrogram
from .diffusion import ARVDiffusion, ARVSampler, VDiffusion, VSampler
from .utils import closest_power_2, default, downsample, exists, groupby, randn_like, upsample

class DiffusionModel(nn.Module):
    def init(
        self,
        net_t: Callable,
        diffusion_t: Callable = VDiffusion,
        sampler_t: Callable = VSampler,
        loss_fn: Callable = torch.nn.functional.mse_loss,
        dim: int = 1,
        **kwargs,
    ):
        super().init()
        diffusion_kwargs, kwargs = groupby(“diffusion_”, kwargs)
        sampler_kwargs, kwargs = groupby(“sampler_”, kwargs)

        self.net = net_t(dim=dim, **kwargs)
        self.diffusion = diffusion_t(net=self.net, loss_fn=loss_fn, **diffusion_kwargs)
        self.sampler = sampler_t(net=self.net, **sampler_kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.diffusion(*args, **kwargs)

    @torch.no_grad()
    def sample(self, *args, **kwargs) -> Tensor:
        return self.sampler(*args, **kwargs)

class EncoderBase(nn.Module, ABC):
    @abstractmethod
    def init(self):
        super().init()
        self.out_channels = None
        self.downsample_factor = None

class AdapterBase(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def decode(self, x: Tensor) -> Tensor:
        pass

class DiffusionAE(DiffusionModel):
    def init(
        self,
        in_channels: int,
        channels: Sequence[int],
        encoder: EncoderBase,
        inject_depth: int,
        latent_factor: Optional[int] = None,
        adapter: Optional[AdapterBase] = None,
        **kwargs,
    ):
        context_channels = [0] * len(channels)
        context_channels[inject_depth] = encoder.out_channels
        super().init(
            in_channels=in_channels,
            channels=channels,
            context_channels=context_channels,
            **kwargs,
        )
        self.in_channels = in_channels
        self.encoder = encoder
        self.inject_depth = inject_depth
        self.latent_factor = default(latent_factor, self.encoder.downsample_factor)
        self.adapter = adapter.requires_grad_(False) if exists(adapter) else None

    def forward(self, x: Tensor, with_info: bool = False, **kwargs) -> Union[Tensor, Tuple[Tensor, Any]]:
        latent, info = self.encode(x, with_info=True)
        channels = [None] * self.inject_depth + [latent]
        x = self.adapter.encode(x) if exists(self.adapter) else x
        loss = super().forward(x, channels=channels, **kwargs)
        return (loss, info) if with_info else loss

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    @torch.no_grad()
    def decode(self, latent: Tensor, generator: Optional[Generator] = None, **kwargs) -> Tensor:
        b = latent.shape[0]
        noise_length = closest_power_2(latent.shape[2] * self.latent_factor)
        noise = torch.randn(
            (b, self.in_channels, noise_length),
            device=latent.device,
            dtype=latent.dtype,
            generator=generator,
        )
        channels = [None] * self.inject_depth + [latent]
        out = super().sample(noise, channels=channels, **kwargs)
        return self.adapter.decode(out) if exists(self.adapter) else out

class DiffusionUpsampler(DiffusionModel):
    def init(
        self,
        in_channels: int,
        upsample_factor: int,
        net_t: Callable,
        **kwargs,
    ):
        self.upsample_factor = upsample_factor
        super().init(
            net_t=AppendChannelsPlugin(net_t, channels=in_channels),
            in_channels=in_channels,
            **kwargs,
        )

    def reupsample(self, x: Tensor) -> Tensor:
        x = x.clone()
        x = downsample(x, factor=self.upsample_factor)
        x = upsample(x, factor=self.upsample_factor)
        return x

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        reupsampled = self.reupsample(x)
        return super().forward(x, *args, append_channels=reupsampled, **kwargs)

    @torch.no_grad()
    def sample(self, downsampled: Tensor, generator: Optional[Generator] = None, **kwargs) -> Tensor:
        reupsampled = upsample(downsampled, factor=self.upsample_factor)
        noise = randn_like(reupsampled, generator=generator)
        return super().sample(noise, append_channels=reupsampled, **kwargs)

class DiffusionVocoder(DiffusionModel):
    def init(
        self,
        net_t: Callable,
        mel_channels: int,
        mel_n_fft: int,
        mel_hop_length: Optional[int] = None,
        mel_win_length: Optional[int] = None,
        in_channels: int = 1,
        **kwargs,
    ):
        mel_hop_length = default(mel_hop_length, floor(mel_n_fft) // 4)
        mel_win_length = default(mel_win_length, mel_n_fft)
        mel_kwargs, kwargs = groupby(“mel_”, kwargs)
        super().init(
            net_t=AppendChannelsPlugin(net_t, channels=1),
            in_channels=1,
            **kwargs,
        )
        self.to_spectrogram = MelSpectrogram(
            n_fft=mel_n_fft,
            hop_length=mel_hop_length,
            win_length=mel_win_length,
            n_mel_channels=mel_channels,
            **mel_kwargs,
        )
        self.to_flat = nn.ConvTranspose1d(
            in_channels=mel_channels,
            out_channels=1,
            kernel_size=mel_win_length,
            stride=mel_hop_length,
            padding=(mel_win_length - mel_hop_length) // 2,
            bias=False,
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        spectrogram = rearrange(self.to_spectrogram(x), “b c f l -> (b c) f l”)
        spectrogram_flat = self.to_flat(spectrogram)
        x = rearrange(x, “b c t -> (b c) 1 t”)
        return super().forward(x, *args, append_channels=spectrogram_flat, **kwargs)

    @torch.no_grad()
    def sample(self, spectrogram: Tensor, generator: Optional[Generator] = None, *kwargs) -> Tensor:
        spectrogram, ps = pack([spectrogram], " f l")
        spectrogram_flat = self.to_flat(spectrogram)
        noise = randn_like(spectrogram_flat, generator=generator)
        waveform = super().sample(noise, append_channels=spectrogram_flat, *kwargs)
        waveform = rearrange(waveform, “… 1 t -> … t”)
        waveform = unpack(waveform, ps, " t")[0]
        return waveform

class DiffusionAR(DiffusionModel):
    def init(
        self,
        in_channels: int,
        length: int,
        num_splits: int,
        diffusion_t: Callable = ARVDiffusion,
        sampler_t: Callable = ARVSampler,
        **kwargs,
    ):
        super().init(
            in_channels=in_channels + 1,
            out_channels=in_channels,
            diffusion_t=diffusion_t,
            diffusion_length=length,
            diffusion_num_splits=num_splits,
            sampler_t=sampler_t,
            sampler_in_channels=in_channels,
            sampler_length=length,
            sampler_num_splits=num_splits,
            use_time_conditioning=False,
            use_modulation=False,
            **kwargs,
        )
