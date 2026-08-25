from thop import profile
import torch
from models.network import UNet


net = UNet(width=64, enc_blk_nums=[4, 3], dec_blk_nums=[2, 2], middle_blk_num=1).cuda()

torch.cuda.reset_max_memory_allocated()
x = torch.rand(1, 3, 512, 512).cuda()
y = net(x)
print(y.shape)
max_memory_reserved = torch.cuda.max_memory_reserved(device='cuda') / (1024 ** 2)
print(f"Model Max Memory Usage: {max_memory_reserved:.2f} MB")

flops, params = profile(net, (x,))
print('FLOPs: %.4f G, Params: %.4f M' % (flops / 1e9, params / 1e6))