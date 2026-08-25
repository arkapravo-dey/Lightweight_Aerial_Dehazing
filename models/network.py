import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# LayerNorm2d

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, -1, 1, 1) * y + bias.view(1, -1, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var, weight = ctx.saved_variables

        g = grad_output * weight.view(1, -1, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)

        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)

        return gx, \
               (grad_output * y).sum(dim=(0,2,3)), \
               grad_output.sum(dim=(0,2,3)), \
               None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0, groups=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size,
                              stride=stride, padding=padding,
                              groups=groups, bias=False)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



class HazeEstimator(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.haze = nn.Conv2d(c, 1, 3, padding=1)

    def forward(self, x):
        return torch.sigmoid(self.haze(x))


class Channel(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1)
        )

        self.query_proj = nn.Conv2d(channels, channels, 1)
        self.key_proj   = nn.Conv2d(channels, channels, 1)
        self.value_proj = nn.Conv2d(1, channels, 1)

    def forward(self, x, haze):

        channel_att = self.mlp(x)          # Key source

        Q = self.query_proj(x)             # Query from features
        K = self.key_proj(channel_att)     # Key from channel attention
        V = self.value_proj(haze)          # Value from haze

        attn = torch.sigmoid(Q * K)        

        return x + attn * V                # Residual refinement

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1),
             torch.mean(x, 1).unsqueeze(1)),
            dim=1
        )

class Spatial(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, 3, padding=1, relu=False)

        self.query_proj = nn.Conv2d(channels, 1, 1)
        self.key_proj   = nn.Conv2d(1, 1, 1)
        self.value_proj = nn.Conv2d(1, channels, 1)

    def forward(self, x, haze):

        x_compress = self.compress(x)

        spatial_att = torch.sigmoid(self.spatial(x_compress))  # Key source

        Q = self.query_proj(x)             # Query
        K = self.key_proj(spatial_att)     # Key
        V = self.value_proj(haze)          # Value

        attn = torch.sigmoid(Q * K)        # Spatial interaction

        return x + attn * V                # Residual refinement


class HazeGuidedAttention(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.haze_estimator = HazeEstimator(c)
        self.ChannelGate = Channel(c)
        self.SpatialGate = Spatial(c)

    def forward(self, x):

        haze = self.haze_estimator(x)

        x = self.ChannelGate(x, haze)
        x = self.SpatialGate(x, haze)

        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class Block(nn.Module):
    def __init__(self, c, drop_path=0., FFN_Expand=2):
        super().__init__()

        self.norm1 = LayerNorm2d(c)

        self.conv_h = nn.Conv2d(c, c, (1,5), padding=(0,2), groups=c)
        self.conv_v = nn.Conv2d(c, c, (5,1), padding=(2,0), groups=c)

        self.conv_7x7 = nn.Conv2d(c, c, 3, padding=3, dilation=3, groups=c)

        self.fuse = nn.Conv2d(c * 3, c, 1)

        self.attn = HazeGuidedAttention(c)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

        self.norm2 = LayerNorm2d(c)
        self.pwconv1 = nn.Conv2d(c, FFN_Expand * c, 1)
        self.act = SimpleGate()
        self.pwconv2 = nn.Conv2d(FFN_Expand * c // 2, c, 1)

    def forward(self, x):

        x_norm = self.norm1(x)

        h = self.conv_h(x_norm)
        v = self.conv_v(x_norm)
        k = self.conv_7x7(x_norm)

        out = torch.cat([h, v, k], dim=1)
        out = self.fuse(out)

        out = self.attn(out)

        out = x + self.drop_path(self.beta * out)

        ffn = self.norm2(out)
        ffn = self.pwconv1(ffn)
        ffn = self.act(ffn)
        ffn = self.pwconv2(ffn)

        return out + self.drop_path(self.gamma * ffn)


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64, patch_size=8):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        return self.proj(x)

class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=64, out_chans=3, patch_size=8):
        super().__init__()
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans,
                                       kernel_size=patch_size,
                                       stride=patch_size)

    def forward(self, x):
        return self.proj(x)

class UNet(nn.Module):
    def __init__(self, img_channel=3, width=64,
                 middle_blk_num=1,
                 enc_blk_nums=[4,3],
                 dec_blk_nums=[1,1],
                 patch_size=8):
        super().__init__()

        self.patch_embed = PatchEmbed(img_channel, width, patch_size)

        self.intro = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU()
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width

        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[Block(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan, 2, 2))

        self.middle_blks = nn.Sequential(*[Block(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.ConvTranspose2d(chan, chan, 2, 2))
            self.decoders.append(nn.Sequential(*[Block(chan) for _ in range(num)]))

        self.patch_unembed = PatchUnEmbed(width, img_channel, patch_size)

    def forward(self, inp):

        x = x_skip = self.intro(inp)
        x = self.patch_embed(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.patch_unembed(x)

        return x + x_skip