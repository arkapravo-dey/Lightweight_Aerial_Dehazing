import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# ============================================================
# LayerNorm2d
# ============================================================

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


# ============================================================
# Simple Gate
# ============================================================

class SimpleGate(nn.Module):

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# ============================================================
# MSPA - Multi-Scale Pixel Attention
# ============================================================

class MSPA(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv3 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
            bias=False
        )

        self.conv5 = nn.Conv2d(
            channels,
            channels,
            5,
            padding=2,
            groups=channels,
            bias=False
        )

        self.fuse = nn.Conv2d(
            channels * 3,
            channels,
            1,
            bias=True
        )

        self.pixel_gate = nn.Conv2d(
            channels,
            1,
            1,
            bias=True
        )

    def forward(self, x):

        x_abs = torch.abs(x)

        x3 = self.conv3(x)
        x5 = self.conv5(x)

        feat = torch.cat(
            [x_abs, x3, x5],
            dim=1
        )

        feat = self.fuse(feat)

        attn = torch.sigmoid(
            self.pixel_gate(feat)
        )

        return x * attn


# ============================================================
# Haar DWT / IWT
# ============================================================

class HaarDWT(nn.Module):

    def forward(self, x):

        H, W = x.shape[-2:]

        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(
                "HaarDWT requires even spatial dimensions."
            )

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        LL = (x00 + x01 + x10 + x11) / 2.0
        LH = (-x00 + x01 - x10 + x11) / 2.0
        HL = (-x00 - x01 + x10 + x11) / 2.0
        HH = (x00 - x01 - x10 + x11) / 2.0

        return LL, LH, HL, HH


class HaarIWT(nn.Module):

    def forward(self, LL, LH, HL, HH):

        x00 = (LL - LH - HL + HH) / 2.0
        x01 = (LL + LH - HL - HH) / 2.0
        x10 = (LL - LH + HL - HH) / 2.0
        x11 = (LL + LH + HL + HH) / 2.0

        B, C, H, W = LL.shape

        out = torch.zeros(
            B,
            C,
            H * 2,
            W * 2,
            device=LL.device,
            dtype=LL.dtype
        )

        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11

        return out


# ============================================================
# FBR - Frequency Band Refinement
# ============================================================

class FBR(nn.Module):

    def __init__(self, channels):
        super().__init__()

        # LL
        self.ll_dw3 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
            bias=False
        )

        self.ll_dw5 = nn.Conv2d(
            channels,
            channels,
            5,
            padding=2,
            groups=channels,
            bias=False
        )

        # LH
        self.lh_h = nn.Conv2d(
            channels,
            channels,
            (1, 5),
            padding=(0, 2),
            groups=channels,
            bias=False
        )

        self.lh_v = nn.Conv2d(
            channels,
            channels,
            (5, 1),
            padding=(2, 0),
            groups=channels,
            bias=False
        )

        # HL
        self.hl_v = nn.Conv2d(
            channels,
            channels,
            (5, 1),
            padding=(2, 0),
            groups=channels,
            bias=False
        )

        self.hl_h = nn.Conv2d(
            channels,
            channels,
            (1, 5),
            padding=(0, 2),
            groups=channels,
            bias=False
        )

        # HH
        self.hh = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
            bias=False
        )

        self.band_fuse = nn.Conv2d(
            channels * 4,
            channels * 4,
            1,
            bias=True
        )

        self.dwt = HaarDWT()
        self.iwt = HaarIWT()

    def forward(self, x):

        LL, LH, HL, HH = self.dwt(x)

        LL_ref = self.ll_dw3(LL)
        LL_ref = self.ll_dw5(LL_ref)
        LL_ref = LL + LL_ref

        LH_ref = self.lh_h(LH)
        LH_ref = self.lh_v(LH_ref)
        LH_ref = LH + LH_ref

        HL_ref = self.hl_v(HL)
        HL_ref = self.hl_h(HL_ref)
        HL_ref = HL + HL_ref

        HH_ref = self.hh(HH)
        HH_ref = HH + HH_ref

        bands = torch.cat(
            [LL_ref, LH_ref, HL_ref, HH_ref],
            dim=1
        )

        bands = self.band_fuse(bands)

        LL_ref, LH_ref, HL_ref, HH_ref = torch.chunk(
            bands,
            4,
            dim=1
        )

        return self.iwt(
            LL_ref,
            LH_ref,
            HL_ref,
            HH_ref
        )


# ============================================================
# CGB - Competitive Gating Block
# ============================================================

class CGB(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.preserve = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                1,
                bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                1,
                bias=False
            )
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
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                1,
                bias=False
            )
        )

        self.gate_p = nn.Conv2d(
            channels,
            channels,
            1
        )

        self.gate_s = nn.Conv2d(
            channels,
            channels,
            1
        )

    def forward(self, x):

        Xp = self.preserve(x)
        Xs = self.suppress(x)

        Ap = self.gate_p(x)
        As = self.gate_s(x)

        gates = torch.stack(
            [Ap, As],
            dim=1
        )

        gates = F.softmax(
            gates,
            dim=1
        )

        Gp = gates[:, 0]
        Gs = gates[:, 1]

        return Gp * Xp + Gs * Xs


# ============================================================
# Proposed Block
# MSPA -> FBR -> CGB -> FFN
# ============================================================

class ProposedBlock(nn.Module):

    def __init__(
        self,
        c,
        drop_path=0.,
        FFN_Expand=2
    ):
        super().__init__()

        self.norm1 = LayerNorm2d(c)

        self.mspa = MSPA(c)
        self.fbr = FBR(c)
        self.cgb = CGB(c)

        self.beta = nn.Parameter(
            torch.zeros(1, c, 1, 1)
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0.
            else nn.Identity()
        )

        self.norm2 = LayerNorm2d(c)

        self.pwconv1 = nn.Conv2d(
            c,
            FFN_Expand * c,
            1
        )

        self.act = SimpleGate()

        self.pwconv2 = nn.Conv2d(
            FFN_Expand * c // 2,
            c,
            1
        )

        self.gamma = nn.Parameter(
            torch.zeros(1, c, 1, 1)
        )

    def forward(self, x):

        out = self.norm1(x)

        out = self.mspa(out)
        out = self.fbr(out)
        out = self.cgb(out)

        x = x + self.drop_path(
            self.beta * out
        )

        ffn = self.norm2(x)

        ffn = self.pwconv1(ffn)
        ffn = self.act(ffn)
        ffn = self.pwconv2(ffn)

        x = x + self.drop_path(
            self.gamma * ffn
        )

        return x


# ============================================================
# Patch Embed
# ============================================================

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


# ============================================================
# Patch UnEmbed
# ============================================================

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


# ============================================================
# U-Net
# ============================================================

class UNet(nn.Module):

    def __init__(
        self,
        img_channel=3,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[4, 3],
        dec_blk_nums=[1, 1],
        patch_size=8
    ):
        super().__init__()

        self.intro = nn.Sequential(
            nn.Conv2d(
                3,
                3,
                3,
                padding=1
            ),
            nn.ReLU()
        )

        self.patch_embed = PatchEmbed(
            img_channel,
            width,
            patch_size
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width

        for num in enc_blk_nums:

            self.encoders.append(
                nn.Sequential(
                    *[
                        ProposedBlock(chan)
                        for _ in range(num)
                    ]
                )
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

        for num in dec_blk_nums:

            self.ups.append(
                nn.ConvTranspose2d(
                    chan,
                    chan,
                    2,
                    2
                )
            )

            self.decoders.append(
                nn.Sequential(
                    *[
                        ProposedBlock(chan)
                        for _ in range(num)
                    ]
                )
            )

        self.patch_unembed = PatchUnEmbed(
            width,
            img_channel,
            patch_size
        )

    def forward(self, inp):

        # Initial feature extraction
        x = self.intro(inp)

        # Patch embedding
        x = self.patch_embed(x)

        # Skip after PatchEmbed
        x_skip = x

        # Encoder
        encs = []

        for encoder, down in zip(
            self.encoders,
            self.downs
        ):
            x = encoder(x)

            encs.append(x)

            x = down(x)

        # Bottleneck
        x = self.middle_blks(x)

        # Decoder
        for decoder, up, enc_skip in zip(
            self.decoders,
            self.ups,
            encs[::-1]
        ):
            x = up(x)

            x = x + enc_skip

            x = decoder(x)

        # Skip before PatchUnEmbed
        x = x + x_skip

        # Patch unembedding
        x = self.patch_unembed(x)

        return x


