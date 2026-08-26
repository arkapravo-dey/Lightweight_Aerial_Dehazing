import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SimpleGate(nn.Module):

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


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

        feat = x_abs + x3 + x5

        attn = torch.sigmoid(
            self.pixel_gate(feat)
        )

        return x * (1.0 + attn)

_HAAR_BASE_KERNELS = torch.tensor([
    [[0.5, 0.5], [0.5, 0.5]],
    [[-0.5, 0.5], [-0.5, 0.5]],
    [[-0.5, -0.5], [0.5, 0.5]],
    [[0.5, -0.5], [-0.5, 0.5]]
])


class HaarDWT(nn.Module):

    def __init__(self, channels):
        super().__init__()

        weight = _HAAR_BASE_KERNELS.unsqueeze(1).repeat(
            channels,
            1,
            1,
            1
        )

        self.register_buffer(
            "weight",
            weight,
            persistent=False
        )

    def forward(self, x):

        B, C, H, W = x.shape

        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(
                "HaarDWT requires even spatial dimensions."
            )

        out = F.conv2d(
            x,
            self.weight.to(dtype=x.dtype),
            stride=2,
            groups=C
        )

        out = out.view(
            B,
            C,
            4,
            H // 2,
            W // 2
        )

        LL = out[:, :, 0]
        LH = out[:, :, 1]
        HL = out[:, :, 2]
        HH = out[:, :, 3]

        return LL, LH, HL, HH


class HaarIWT(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, LL, LH, HL, HH):

        x00 = (LL - LH - HL + HH) / 2.0
        x01 = (LL + LH - HL - HH) / 2.0
        x10 = (LL - LH + HL - HH) / 2.0
        x11 = (LL + LH + HL + HH) / 2.0

        B, C, H, W = LL.shape

        x = torch.stack(
            [x00, x01, x10, x11],
            dim=-1
        )

        x = x.view(
            B,
            C,
            H,
            W,
            2,
            2
        )

        x = x.permute(
            0,
            1,
            2,
            4,
            3,
            5
        )

        x = x.reshape(
            B,
            C,
            H * 2,
            W * 2
        )

        return x

class FBR(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.dwt = HaarDWT(channels)
        self.iwt = HaarIWT()

        self.ll_dw = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
            bias=False
        )

        self.ll_pw = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False
        )

        self.ll_gate = nn.Conv2d(
            channels,
            1,
            1,
            bias=True
        )

        self.high_dw = nn.Conv2d(
            channels,
            channels,
            3,
            padding=2,
            dilation=2,
            groups=channels,
            bias=False
        )

        self.high_pw = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False
        )

        self.high_gate = nn.Conv2d(
            channels,
            1,
            1,
            bias=True
        )

    def forward(self, x):

        LL, LH, HL, HH = self.dwt(x)

        ll_feat = self.ll_dw(LL)
        ll_feat = self.ll_pw(ll_feat)

        ll_attn = torch.sigmoid(
            self.ll_gate(ll_feat)
        )

        LL = LL * (1.0 + ll_attn)

        high = torch.stack(
            [LH, HL, HH],
            dim=1
        )

        B, N, C, H, W = high.shape

        high = high.reshape(
            B * N,
            C,
            H,
            W
        )

        high_ref = self.high_dw(high)
        high_ref = self.high_pw(high_ref)

        high = high + high_ref

        high_attn = torch.sigmoid(
            self.high_gate(high)
        )

        high = high * (1.0 + high_attn)

        high = high.reshape(
            B,
            N,
            C,
            H,
            W
        )

        LH = high[:, 0]
        HL = high[:, 1]
        HH = high[:, 2]

        return self.iwt(
            LL,
            LH,
            HL,
            HH
        )


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

        x = self.intro(inp)

        x = self.patch_embed(x)

        x_skip = x

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

        x = x + x_skip

        x = self.patch_unembed(x)

        return x