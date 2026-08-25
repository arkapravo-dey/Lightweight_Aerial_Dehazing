import torch
import numpy as np
import cv2

from utils.meters import AverageMeter

def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images.
    Input images must be uint8 numpy arrays with shape HxW or HxWx3.
    """
    def _ssim_single_channel(x, y, data_range=255):
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        x = x.astype(np.float64)
        y = y.astype(np.float64)

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu_x = cv2.filter2D(x, -1, window)[5:-5, 5:-5]
        mu_y = cv2.filter2D(y, -1, window)[5:-5, 5:-5]

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy   = mu_x * mu_y

        sigma_x_sq = cv2.filter2D(x * x, -1, window)[5:-5, 5:-5] - mu_x_sq
        sigma_y_sq = cv2.filter2D(y * y, -1, window)[5:-5, 5:-5] - mu_y_sq
        sigma_xy   = cv2.filter2D(x * y, -1, window)[5:-5, 5:-5] - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        return ssim_map.mean()

    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    if img1.ndim == 2:  # grayscale
        return _ssim_single_channel(img1, img2)
    elif img1.ndim == 3 and img1.shape[2] == 3:  # color
        ssims = []
        for i in range(3):
            ssims.append(_ssim_single_channel(img1[..., i], img2[..., i]))
        return np.mean(ssims)
    else:
        raise ValueError("Wrong input dimensions for SSIM. Expected HxW or HxWx3.")


def calculate_PSNR_SSIM(val_loader, network):
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():
            output = network(source_img).clamp_(-1, 1)

        output = output * 0.5 + 0.5
        target_img = target_img * 0.5 + 0.5

        # Convert to numpy
        output = output.permute(0, 2, 3, 1).cpu().numpy()  # (B,H,W,C)
        target_img = target_img.permute(0, 2, 3, 1).cpu().numpy()

        for out_img, tgt_img in zip(output, target_img):
            out_img = (out_img * 255.0).astype(np.uint8)
            tgt_img = (tgt_img * 255.0).astype(np.uint8)

            # PSNR
            psnr_val = cv2.PSNR(out_img, tgt_img)
            psnr_meter.update(psnr_val, 1)

            # SSIM
            ssim_val = calculate_ssim(out_img, tgt_img)
            ssim_meter.update(ssim_val, 1)

    return psnr_meter.avg, ssim_meter.avg
