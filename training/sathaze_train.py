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
from datasets.sathaze import SateHaze1KDataset, transform
from models.network import UNet


def train(train_loader, network, criterion, optimizer, scaler, epoch):

    losses = AverageMeter()

    torch.cuda.empty_cache()
    network.train()

    batch_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch} - Training",
        leave=False
    )

    for batch_idx, batch in enumerate(batch_progress):

        source_img = batch['source'].cuda(non_blocking=True)
        target_img = batch['target'].cuda(non_blocking=True)

        with autocast():

            output = network(source_img)
            loss = criterion(output, target_img)

        losses.update(loss.item())

        optimizer.zero_grad(set_to_none=True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return losses.avg


if __name__ == '__main__':

    args = get_args()

    network = UNet(
        width=64,
        enc_blk_nums=[4, 3],
        dec_blk_nums=[2, 2],
        middle_blk_num=1
    ).cuda()

    network = nn.DataParallel(network).cuda()

    criterion = nn.L1Loss()

    optimizer_type = args.optimizer.lower()

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=args.lr
        )

    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=args.lr
        )

    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.eta_min
    )

    scaler = torch.cuda.amp.GradScaler()

    train_dataset = SateHaze1KDataset(
        root_dir=args.train_dir,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_thick_dataset = SateHaze1KDataset(
        root_dir=args.val_thick_dir,
        transform=transform
    )

    val_moderate_dataset = SateHaze1KDataset(
        root_dir=args.val_moderate_dir,
        transform=transform
    )

    val_thin_dataset = SateHaze1KDataset(
        root_dir=args.val_thin_dir,
        transform=transform
    )

    val_thick_loader = DataLoader(
        val_thick_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_moderate_loader = DataLoader(
        val_moderate_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_thin_loader = DataLoader(
        val_thin_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(
        args.save_dir,
        args.model + '.pth'
    )

    print(
        '==> Start training, current model name: '
        + args.model
    )

    print(f"Train samples : {len(train_dataset)}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Num workers   : {args.num_workers}")
    print(f"GPUs          : {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(
            f"GPU {i}: "
            f"{torch.cuda.get_device_name(i)}"
        )

    writer = SummaryWriter(
        log_dir=os.path.join(
            args.log_dir,
            args.model
        )
    )

    best_psnr = 0
    best_ssim = 0

    for epoch in tqdm(
        range(args.epochs + 1),
        desc="Training Progress",
        leave=True
    ):

        loss = train(
            train_loader,
            network,
            criterion,
            optimizer,
            scaler,
            epoch
        )

        writer.add_scalar(
            'train_loss',
            loss,
            epoch
        )

        scheduler.step()

        if epoch % args.eval_freq == 0:

            psnr_thick, ssim_thick = calculate_PSNR_SSIM(
                val_thick_loader,
                network
            )

            psnr_moderate, ssim_moderate = calculate_PSNR_SSIM(
                val_moderate_loader,
                network
            )

            psnr_thin, ssim_thin = calculate_PSNR_SSIM(
                val_thin_loader,
                network
            )

            avg_psnr = (
                psnr_thick
                + psnr_moderate
                + psnr_thin
            ) / 3

            avg_ssim = (
                ssim_thick
                + ssim_moderate
                + ssim_thin
            ) / 3

            writer.add_scalar(
                'valid_psnr_thick',
                psnr_thick,
                epoch
            )

            writer.add_scalar(
                'valid_psnr_moderate',
                psnr_moderate,
                epoch
            )

            writer.add_scalar(
                'valid_psnr_thin',
                psnr_thin,
                epoch
            )

            writer.add_scalar(
                'valid_psnr_avg',
                avg_psnr,
                epoch
            )

            writer.add_scalar(
                'valid_ssim_avg',
                avg_ssim,
                epoch
            )

            tqdm.write(
                f"Epoch {epoch}, "
                f"Loss: {loss:.4f}, "
                f"PSNR Thick: {psnr_thick:.2f}, "
                f"PSNR Moderate: {psnr_moderate:.2f}, "
                f"PSNR Thin: {psnr_thin:.2f}, "
                f"Avg PSNR: {avg_psnr:.2f}, "
                f"Avg SSIM: {avg_ssim:.2f}"
            )

            if avg_psnr > best_psnr:

                best_psnr = avg_psnr

                torch.save(
                    {
                        'state_dict':
                        network.state_dict()
                    },
                    model_path.replace(
                        ".pth",
                        "_bestPSNR.pth"
                    )
                )

                writer.add_scalar(
                    'best_psnr',
                    best_psnr,
                    epoch
                )

                tqdm.write(
                    f"New Best PSNR: "
                    f"{best_psnr:.2f} "
                    f"at Epoch {epoch}"
                )

            if avg_ssim > best_ssim:

                best_ssim = avg_ssim

                torch.save(
                    {
                        'state_dict':
                        network.state_dict()
                    },
                    model_path.replace(
                        ".pth",
                        "_bestSSIM.pth"
                    )
                )

                writer.add_scalar(
                    'best_ssim',
                    best_ssim,
                    epoch
                )

                tqdm.write(
                    f"New Best SSIM: "
                    f"{best_ssim:.4f} "
                    f"at Epoch {epoch}"
                )

        if epoch % args.checkpoint_freq == 0:

            checkpoint_path = os.path.join(
                save_dir,
                f'{args.model}_epoch{epoch}.pth'
            )

            torch.save(
                {
                    'state_dict':
                    network.state_dict()
                },
                checkpoint_path
            )