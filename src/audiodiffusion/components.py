from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from a_unet import *
from a_unet.apex import *
from einops import pack, unpack
from torch import Tensor, nn
from torchaudio import transforms

def UNetV0(
    dim: int,
    in_channels: int,
    channels: Sequence[int],
    factors: Sequence[int],
    items: Sequence[int],
    attentions: Optional[Sequence[int]] = None,
    cross_attentions: Optional[Sequence[int]] = None,
    context_channels: Optional[Sequence[int]] = None,
    attention_features: Optional[int] = None,
    attention_heads: Optional[int] = None,
    embedding_features: Optional[int] = None,
    resnet_groups: int = 8,
    use_modulation: bool = True,
    modulation_features: int = 1024,
    embedding_max_length: Optional[int] = None,
    use_time_conditioning: bool = True,
    use_embedding_cfg: bool = False,
    use_text_conditioning: bool = False,
    out_channels: Optional[int] = None,
):
    num_layers = len(channels)
    attentions = default(attentions, [0] * num_layers)
    cross_attentions = default(cross_attentions, [0] * num_layers)
    context_channels = default(context_channels, [0] * num_layers)

    UNetV0 = XUNet

    if use_embedding_cfg:
        assert exists(embedding_max_length), “use_embedding_cfg requires embedding_max_length”
        UNetV0 = ClassifierFreeGuidancePlugin(UNetV0, embedding_max_length)

    if use_text_conditioning:
        UNetV0 = TextConditioningPlugin(UNetV0)

    if use_time_conditioning:
        assert use_modulation, “use_time_conditioning requires use_modulation=True”
        UNetV0 = TimeConditioningPlugin(UNetV0)

    return UNetV0(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        blocks=[
            XBlock(
                channels=channels,
                factor=factor,
                context_channels=ctx_channels,
                items=(
                    [ResnetItem]
                    + [ModulationItem] * use_modulation
                    + [InjectChannelsItem] * (ctx_channels > 0)
                    + [AttentionItem] * att
                    + [CrossAttentionItem] * cross
                )
                * items,
            )
            for channels, factor, items, att, cross, ctx_channels in zip(*xs)
        ],
        skip_t=SkipModulate if use_modulation else SkipCat,
        attention_features=attention_features,
        attention_heads=attention_heads,
        embedding_features=embedding_features,
        modulation_features=modulation_features,
        resnet_groups=resnet_groups,
    )

def LTPlugin(
    net_t: Callable, num_filters: int, window_length: int, stride: int
) -> Callable[…, nn.Module]:
  
    def Net(
        dim: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        out_channels = default(out_channels, in_channels)
        in_channel_transform = in_channels * num_filters
        out_channel_transform = out_channels * num_filters  
      
        padding = window_length // 2 - stride // 2
        encode = Conv(
            dim=dim,
            in_channels=in_channels,
            out_channels=in_channel_transform,
            kernel_size=window_length,
            stride=stride,
            padding=padding,
            padding_mode=“reflect”,
            bias=False,
        )
        decode = nn.ConvTranspose1d(
            in_channels=out_channel_transform,
            out_channels=out_channels,
            kernel_size=window_length,
            stride=stride,
            padding=padding,
            bias=False,
        )
        net = net_t(
            dim=dim,
            in_channels=in_channel_transform,
            out_channels=out_channel_transform,
            **kwargs
        )

        def forward(x: Tensor, *args, **kwargs):
            x = encode(x)
            x = net(x, *args, **kwargs)
            x = decode(x)
            return x

        return Module([encode, decode, net], forward)

    return Net

def AppendChannelsPlugin(
    net_t: Callable, channels: int
):
    def Net(
        in_channels: int, out_channels: Optional[int] = None, **kwargs
    ) -> nn.Module:
        out_channels = default(out_channels, in_channels)
        net = net_t(
            in_channels=in_channels + channels, out_channels=out_channels, **kwargs
        )

        def forward(x: Tensor, *args, append_channels: Tensor, **kwargs):
            x = torch.cat([x, append_channels], dim=1)
            return net(x, *args, kwargs)

        return Module([net], forward)

    return Net

class MelSpectrogram(nn.Module):
    def init(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        sample_rate: int,
        n_mel_channels: int,
        center: bool = False,
        normalize: bool = False,
        normalize_log: bool = False,
    ):
        super().init()
        self.padding = (n_fft - hop_length) // 2
        self.normalize = normalize
        self.normalize_log = normalize_log
        self.hop_length = hop_length

        self.to_spectrogram = transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=center,
            power=None,
        )

        self.to_mel_scale = transforms.MelScale(
            n_mels=n_mel_channels, n_stft=n_fft // 2 + 1, sample_rate=sample_rate
        )

    def forward(self, waveform: Tensor) -> Tensor:
        waveform, ps = pack([waveform], " t")
        waveform = F.pad(waveform, [self.padding] * 2, mode=“reflect”)
        spectrogram = self.to_spectrogram(waveform)
        spectrogram = torch.abs(spectrogram)
        mel_spectrogram = self.to_mel_scale(spectrogram)
      
        if self.normalize:
            mel_spectrogram = mel_spectrogram / torch.max(mel_spectrogram)
            mel_spectrogram = 2 * torch.pow(mel_spectrogram, 0.25) - 1
        if self.normalize_log:
            mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
      
        return unpack(mel_spectrogram, ps, " f l")[0]
