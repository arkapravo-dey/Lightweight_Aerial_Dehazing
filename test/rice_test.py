import torch
from torch import nn

from models.network import UNet
from utils.metrics import calculate_PSNR_SSIM
from datasets.rice import RICEDataset, transform
from torch.utils.data import DataLoader
from configs.config import get_args

if __name__ == '__main__':
    args = get_args()

    network = UNet(width=64, enc_blk_nums=[4,3], dec_blk_nums=[2, 2], middle_blk_num=1).cuda()
    network = nn.DataParallel(network).cuda()

    val_dataset = RICEDataset(root_dir=args.rice_val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    checkpoint = torch.load(args.load_model_path, map_location='cuda:0')
    network.load_state_dict(checkpoint['state_dict'])
    network.eval()

    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    PSNR, SSIM = calculate_PSNR_SSIM(val_loader, network)

    print(f"PSNR on RICE Val Set: {PSNR:.2f} dB")
    print(f"SSIM on RICE Val Set: {SSIM:.4f}")

