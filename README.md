# Efficient Aerial Image Dehazing via Haze Region Aware Feature Learning and Hadamard Gating

Official PyTorch implementation of the IEEE Geoscience and Remote Sensing Letters (GRSL) paper:

**Efficient Aerial Image Dehazing via Haze Region Aware Feature Learning and Hadamard Gating**
 
**Shiladitya Mondal, Sobhan Kanti Dhara, Anusha Vupputuri**
 
*IEEE Geoscience and Remote Sensing Letters (GRSL), 2026*

---

## Abstract

Remote sensing imagery is highly susceptible to haze, which can obscure visibility and limit the reliability of downstream analysis tasks, making aerial image dehazing critical for space and defense applications. Existing methods often fail to faithfully restore structural details and color fidelity under spatially varying dense haze and are parameter-intensive, limiting their deployment in resource-constrained environments. To address this issue, we propose a lightweight encoder–decoder framework for aerial image dehazing that jointly models anisotropic spatial characteristics and global contextual dependencies. The proposed Multi-scale Directional Feature Fusion (MDFF) module explicitly encodes directional and enlarged receptive-field interactions to capture spatially varying haze distribution, while the Haze Region Aware Refinement (HRAR) module estimates haze regions and refines degraded features through dual-stage attention for guided feature learning. Additionally, the Hadamard-Gated Feature Modulation (HGFM) module introduces parameter-efficient multiplicative feature recalibration to enhance fine textural details. Experimental results on benchmark datasets demonstrate that the proposed method achieves superior restoration performance while maintaining substantially fewer parameters and higher computational efficiency.

---

# Repository Structure

```text
Code/
├── configs/
├── datasets/
├── models/
├── training/
│   ├── rice_train.py
│   └── sathaze_train.py
├── test/
│   ├── rice_test.py
│   └── sathaze_test.py
├── utils/
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Shiladityagit/Lightweight_Aerial_Dehazing.git
cd Lightweight_Aerial_Dehazing
```

---

# Pretrained Models

Download the pretrained models from **[Google Drive](https://drive.google.com/drive/folders/1hyDF1AqQmbrNxOS8UcjXdWEPT45cATHz?usp=sharing)**.

After downloading, place the checkpoint files inside:

```text
checkpoints/
├── GRSL_sathaze_model.pth
└── GRSL_rice_model.pth
```

---

# Training

## Train on SateHaze1K

```bash
python -m training.sathaze_train \
    --train_dir /path/to/Sathaze1k/train \
    --val_thick_dir /path/to/Sathaze1k/val_thick \
    --val_moderate_dir /path/to/Sathaze1k/val_moderate \
    --val_thin_dir /path/to/Sathaze1k/val_thin
```

## Train on RICE1

```bash
python -m training.rice_train \
    --rice_train_dir /path/to/RICE1_split/train \
    --rice_val_dir /path/to/RICE1_split/test
```

---

# Evaluation

## Evaluate on SateHaze1K

```bash
python -m test.sathaze_test \
    --test_thick_dir /path/to/Sathaze1k/test_thick \
    --test_moderate_dir /path/to/Sathaze1k/test_moderate \
    --test_thin_dir /path/to/Sathaze1k/test_thin \
    --load_model_path checkpoints/GRSL_sathaze_model.pth
```

## Evaluate on RICE1

```bash
python -m test.rice_test \
    --rice_val_dir /path/to/RICE1_split/test \
    --load_model_path checkpoints/GRSL_rice_model.pth
```

---

# Citation

If you find this work useful in your research, please cite:

```bibtex
@ARTICLE{11641641,
  author={Mondal, Shiladitya and Dhara, Sobhan Kanti and Vupputuri, Anusha},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Efficient Aerial Image Dehazing via Haze Region Aware Feature Learning and Hadamard Gating}, 
  year={2026},
  volume={23},
  number={},
  pages={6016905-6016905},
  keywords={Modeling;Image dehazing;Modules (abstract algebra);Convolution;PSNR;Modulation;Transformers;Remote sensing;Imaging;Aerial image dehazing;Hadamard gating;Haze region aware refinement (HRAR);remote sensing},
  doi={10.1109/LGRS.2026.3719865}}
```
