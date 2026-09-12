import torch
import torch.nn as nn
from timm.models.layers import DropPath


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, -1, 1, 1) * y
        y = y + bias.view(1, -1, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var, weight = ctx.saved_tensors

        g = grad_output * weight.view(1, -1, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)

        gx = (
            1.0 / torch.sqrt(var + eps)
            * (g - y * mean_gy - mean_g)
        )

        return (
            gx,
            (grad_output * y).sum(dim=(0, 2, 3)),
            grad_output.sum(dim=(0, 2, 3)),
            None
        )


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(
            x,
            self.weight,
            self.bias,
            self.eps
        )

class MSPA(nn.Module):
    def __init__(self, channels, N=3):
        super().__init__()

        self.N = N

        self.conv_1xN = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, N),
            padding=(0, N // 2),
            bias=False
        )

        self.conv_Nx1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(N, 1),
            padding=(N // 2, 0),
            bias=False
        )

        self.conv_d1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            bias=False
        )

        self.conv_d2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            bias=False
        )

        self.conv_d3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=3,
            dilation=3,
            bias=False
        )
        self.fusion = nn.Conv2d(
            channels * 3,
            channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        x = self.conv_1xN(x)
        x = self.conv_Nx1(x)
        x_d1 = self.conv_d1(x)
        x_d2 = self.conv_d2(x)
        x_d3 = self.conv_d3(x)
        x_cat = torch.cat(
            [x_d1, x_d2, x_d3],
            dim=1
        )
        attn = self.fusion(x_cat)
        attn = torch.sigmoid(attn)
        return attn

class FBR(nn.Module):
    def __init__(
        self,
        channels,
        low_ratio=0.25,
        high_ratio=0.60
    ):
        super().__init__()

        if not (0.0 < low_ratio < high_ratio < 1.0):
            raise ValueError(
                "Require 0 < low_ratio < high_ratio < 1."
            )

        self.channels = channels
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio

        self.low_gate = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.Sigmoid()
        )

        self.mid_gate = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.Sigmoid()
        )

        self.high_gate = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.Sigmoid()
        )

    def _get_radial_masks(self, H, W, device, dtype):

        y = torch.arange(
            H,
            device=device,
            dtype=dtype
        ) - (H // 2)

        x = torch.arange(
            W,
            device=device,
            dtype=dtype
        ) - (W // 2)

        yy, xx = torch.meshgrid(
            y,
            x,
            indexing="ij"
        )

        radius = torch.sqrt(
            xx.pow(2) + yy.pow(2)
        )

        max_radius = radius.max().clamp_min(1.0)

        normalized_radius = radius / max_radius

        low_mask = (
            normalized_radius <= self.low_ratio
        )

        mid_mask = (
            (normalized_radius > self.low_ratio)
            & (normalized_radius <= self.high_ratio)
        )

        high_mask = (
            normalized_radius > self.high_ratio
        )

        return (
            low_mask.to(dtype=dtype),
            mid_mask.to(dtype=dtype),
            high_mask.to(dtype=dtype)
        )

    def forward(self, x):

        B, C, H, W = x.shape

        X = torch.fft.fft2(
            x,
            dim=(-2, -1),
            norm="ortho"
        )

        X_shifted = torch.fft.fftshift(
            X,
            dim=(-2, -1)
        )

        magnitude = torch.abs(X_shifted)
        phase = torch.angle(X_shifted)

        low_mask, mid_mask, high_mask = self._get_radial_masks(
            H,
            W,
            x.device,
            magnitude.dtype
        )

        low_mask = low_mask.view(1, 1, H, W)
        mid_mask = mid_mask.view(1, 1, H, W)
        high_mask = high_mask.view(1, 1, H, W)

        M_low = magnitude * low_mask
        M_mid = magnitude * mid_mask
        M_high = magnitude * high_mask

        G_low = self.low_gate(M_low)
        G_mid = self.mid_gate(M_mid)
        G_high = self.high_gate(M_high)

        M_low_ref = M_low * G_low
        M_mid_ref = M_mid * G_mid
        M_high_ref = M_high * G_high

        magnitude_refined = (
            M_low_ref
            + M_mid_ref
            + M_high_ref
        )

        X_refined_shifted = torch.polar(
            magnitude_refined,
            phase
        )

        X_refined = torch.fft.ifftshift(
            X_refined_shifted,
            dim=(-2, -1)
        )

        out = torch.fft.ifft2(
            X_refined,
            dim=(-2, -1),
            norm="ortho"
        )

        return out.real



class CGB(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.preserve = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False
        )

        self.suppress = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False
            ),
            nn.ReLU(inplace=True)
        )

        self.gate = nn.Conv2d(
            channels,
            1,
            1,
            bias=True
        )

    def forward(self, x):

        Xp = self.preserve(x)
        Xs = self.suppress(x)

        Gp = torch.sigmoid(
            self.gate(x)
        )

        Gs = 1.0 - Gp

        return (
            Gp * Xp +
            Gs * Xs
        )


class ProposedBlock(nn.Module):

    def __init__(
        self,
        c,
        drop_path=0.
    ):
        super().__init__()

        self.norm1 = LayerNorm2d(c)

        self.mspa = MSPA(c)
        self.fbr = FBR(c)
        self.cgb = CGB(c)

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0.
            else nn.Identity()
        )

        self.beta = nn.Parameter(
            torch.zeros((1, c, 1, 1))
        )

    def forward(self, x):

        x_norm = self.norm1(x)
        attn = self.mspa(x_norm)
        out = x_norm * attn
        out = self.fbr(out)
        out = self.cgb(out)
        out = x + self.drop_path(
            self.beta * out
        )

        return out

class PatchEmbed(nn.Module):

    def __init__(
        self,
        in_chans=3,
        embed_dim=64,
        patch_size=8
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        return self.proj(x)


class PatchUnEmbed(nn.Module):

    def __init__(
        self,
        embed_dim=64,
        out_chans=3,
        patch_size=8
    ):
        super().__init__()

        self.proj = nn.ConvTranspose2d(
            embed_dim,
            out_chans,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        return self.proj(x)


class UNet(nn.Module):

    def __init__(
        self,
        img_channel=3,
        width=48,
        middle_blk_num=1,
        enc_blk_nums=[3, 2],
        dec_blk_nums=[1, 1],
        patch_size=8
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_channel,
            width,
            patch_size
        )

        self.intro = nn.Sequential(
            nn.Conv2d(
                3,
                3,
                3,
                padding=1
            ),
            nn.ReLU()
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width

        for stage_idx, num in enumerate(enc_blk_nums):

            blocks = []

            for _ in range(num):

                blocks.append(
                    ProposedBlock(chan)
                )

            self.encoders.append(
                nn.Sequential(*blocks)
            )

            self.downs.append(
                nn.Conv2d(
                    chan,
                    chan,
                    2,
                    2
                )
            )

        self.middle_blks = nn.Sequential(
            *[
                ProposedBlock(chan)
                for _ in range(middle_blk_num)
            ]
        )

        for stage_idx, num in enumerate(dec_blk_nums):

            self.ups.append(
                nn.ConvTranspose2d(
                    chan,
                    chan,
                    2,
                    2
                )
            )

            blocks = []

            for _ in range(num):

                blocks.append(
                    ProposedBlock(chan)
                )

            self.decoders.append(
                nn.Sequential(*blocks)
            )

        self.patch_unembed = PatchUnEmbed(
            width,
            img_channel,
            patch_size
        )

    def forward(self, inp):

        x = x_skip = self.intro(inp)
        x = self.patch_embed(x)
        encs = []

        for encoder, down in zip(
            self.encoders,
            self.downs
        ):

            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(
            self.decoders,
            self.ups,
            encs[::-1]
        ):

            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.patch_unembed(x)

        return x + x_skip