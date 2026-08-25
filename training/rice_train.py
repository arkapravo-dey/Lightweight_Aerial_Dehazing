import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from torch.cuda.amp import autocast

from configs.config import get_args
from utils.metrics import calculate_PSNR_SSIM
from utils.meters import AverageMeter
from datasets.rice import RICEDataset, transform
from models.network import UNet

def train(train_loader, network, criterion, optimizer, scaler):
    losses = AverageMeter()

    torch.cuda.empty_cache()
    network.train()

    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with autocast():
            output = network(source_img)
            loss = criterion(output, target_img)

        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg


if __name__ == '__main__':

    args = get_args()

    network = UNet(width=64, enc_blk_nums=[4, 3], dec_blk_nums=[2, 2], middle_blk_num=1).cuda()
    network = nn.DataParallel(network).cuda()

    criterion = nn.L1Loss()

    optimizer_type = args.optimizer.lower()
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    train_dataset = RICEDataset(root_dir=args.rice_train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_dataset = RICEDataset(root_dir=args.rice_val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
  
    model_path = os.path.join(args.save_dir, args.model + '.pth')
    print('==> Start training, current model name: ' + args.model)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.model))

    best_psnr = 0
    best_ssim = 0
    
    for epoch in tqdm(range(args.epochs + 1), desc="Training Progress", leave=True):
        loss = train(train_loader, network, criterion, optimizer, scaler)
        writer.add_scalar('train_loss', loss, epoch)

        scheduler.step()

        if epoch % args.eval_freq == 0:
            avg_psnr, avg_ssim = calculate_PSNR_SSIM(val_loader, network)

            writer.add_scalar('valid_psnr_avg', avg_psnr, epoch)
            writer.add_scalar('valid_psnr_avg', avg_ssim, epoch)

            tqdm.write(f"Epoch {epoch}, Loss: {loss:.4f}, Avg PSNR: {avg_psnr:.2f}, Avg SSIM: {avg_ssim:.2f}")
           
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({'state_dict': network.state_dict()}, model_path.replace(".pth", "_bestPSNR.pth"))
                writer.add_scalar('best_psnr', best_psnr, epoch)
                tqdm.write(f" New Best PSNR: {best_psnr:.2f} at Epoch {epoch}")
        
            if avg_ssim > best_ssim:
                best_ssim = avg_ssim
                torch.save({'state_dict': network.state_dict()}, model_path.replace(".pth", "_bestSSIM.pth"))
                writer.add_scalar('best_ssim', best_ssim, epoch)
                tqdm.write(f" New Best SSIM: {best_ssim:.4f} at Epoch {epoch}")

        if epoch % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(save_dir, f'{args.model}_epoch{epoch}.pth')
            torch.save({'state_dict': network.state_dict()}, checkpoint_path)