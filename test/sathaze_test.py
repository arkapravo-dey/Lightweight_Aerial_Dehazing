import torch
from torch import nn

from models.network import UNet
from utils.metrics import calculate_PSNR_SSIM
from datasets.sathaze import SateHaze1KDataset, transform
from torch.utils.data import DataLoader
from configs.config import get_args

if __name__ == '__main__':
    args = get_args()

    network = UNet(width=64, enc_blk_nums=[4,3], dec_blk_nums=[2, 2], middle_blk_num=1).cuda()
    network = nn.DataParallel(network).cuda()

    test_thick_dataset = SateHaze1KDataset(root_dir=args.test_thick_dir, transform=transform)
    test_moderate_dataset = SateHaze1KDataset(root_dir=args.test_moderate_dir, transform=transform)
    test_thin_dataset = SateHaze1KDataset(root_dir=args.test_thin_dir, transform=transform)

    test_thick_loader = DataLoader(test_thick_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_moderate_loader = DataLoader(test_moderate_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_thin_loader = DataLoader(test_thin_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    checkpoint = torch.load(args.load_model_path, map_location='cuda:0')
    network.load_state_dict(checkpoint['state_dict'])
    network.eval()

    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    PSNR, SSIM = calculate_PSNR_SSIM(test_thin_loader, network)

    print(f"PSNR on Thin Test Set: {PSNR:.2f} dB")
    print(f"SSIM on Thin Test Set: {SSIM:.4f}")


    PSNR, SSIM = calculate_PSNR_SSIM(test_thick_loader, network)

    print(f"PSNR on Thick Test Set: {PSNR:.2f} dB")
    print(f"SSIM on Thick Test Set: {SSIM:.4f}")


    PSNR, SSIM = calculate_PSNR_SSIM(test_moderate_loader, network)

    print(f"PSNR on Moderate Test Set: {PSNR:.2f} dB")
    print(f"SSIM on Moderate Test Set: {SSIM:.4f}")
