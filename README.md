# Feature Fusion CGAN Based HRRP Denosing and Reconstruction :recycle:
This repository provides a comprehensive deep learning framework for High-Resolution Range Profile (HRRP) signal denoising, supporting multiple models and PSNR-controlled training and testing.

> This paper addresses the issue of High Resolution Range Profile (HRRP) data-based Radar Automatic Target Recognition (RATR) under noise interference by proposing a denoising and reconstruction method based on a feature fusion Conditional Generative Adversarial Network (CGAN). Compared with current methods based on Auto-Encoder (AE) models that only achieves local precision, the proposed the CGAN effectively learn the global distribution of HRRP data through the adversarial training of a generator structured as an encoder-decoder and a discriminator composed by a Multilayer Perceptron (MLP). Additionally, to realize precise HRRP denoising and reconstruction, inspired by the application of radial length for rough target classification, we introduce two simple but innovative modules that designed to extract high-dimensional representations of geometry information and identity information, which is finally fused with high-dimensional representation of HRRP extracted by the encoder and serves as the decoder's input. In our experiments, we employs a One Dimensional Convolutional Neural Network (1-D CNN) to classify the denoised and reconstructed HRRPs and evaluate the effectiveness of the proposed method. \textcolor{blue}{Results prove that in the conditions of Peak Signal-to-Noise Ratio (PSNR) 20dB, 10dB and 5dB, the improvement of recognition accuracy, PSNR, and Structural Similarity (SSIM) surpass other methods on both simulated and measured datasets.

<p align="center">
  <img src="method.jpg" width="60%">
</p>

---

## Platform :pushpin:
Developed and tested on PyCharm IDE with Conda environment. Recommended OS:
- Ubuntu 20.04+ 
- Windows 10/11 (WSL2 recommended)

---

## Dependencies :wrench:
```angular2html
conda create -n hrrp_denosing python=3.8
conda activate mlgnn
pip install torch==2.2.0 torchvision==0.17.0
pip install numpy, matplotlib, scikit-image, pandas, seaborn
```

---

## Dataset Structure :file_folder:
Prepare your data with following structure:
```bash
datasets/
├── simulated_3/
│   ├── train/
│   └── train/
└── measured_3/
    ├── train/
    └── test/
```

---

## Quick Start :rocket:

### Training Models

Use the unified training script `train_all.py` to train any model:

```bash
# Train CGAN model at multiple PSNR levels
python train_all.py --model cgan --psnr_levels 20 10 0 --train_dir datasets/simulated_3/train --save_dir checkpoints --batch_size 64 --epochs 200 --save_samples

# Train CAE model at specific PSNR level
python train_all.py --model cae --psnr_levels 10 --train_dir datasets/simulated_3/train --save_dir checkpoints --batch_size 64 --epochs 200

# Train feature extractor modules first (required for CGAN)
python train_all.py --model modules --train_dir datasets/simulated_3/train --save_dir checkpoints/modules --batch_size 256 --epochs 1000

# Train all models at once
python train_all.py --model all --psnr_levels 20 10 0 --train_dir datasets/simulated_3/train --save_dir checkpoints --batch_size 64 --epochs 200
```

### Testing and Comparing Models

Use the unified testing script `test_all.py` to evaluate and compare models:

```bash
# Test CGAN model
python test_all.py --model cgan --psnr_levels 20 10 0 --test_dir datasets/simulated_3/test --cgan_dir checkpoints/cgan --output_dir results --num_samples 10

# Compare all models
python test_all.py --model all --psnr_levels 20 10 0 --test_dir datasets/simulated_3/test --cgan_dir checkpoints/cgan --cae_dir checkpoints/cae --ae_dir checkpoints/ae --output_dir results/comparison --num_samples 10

# Create detailed visualizations
python test_all.py --model all --psnr_levels 10 --test_dir datasets/simulated_3/test --cgan_dir checkpoints/cgan --cae_dir checkpoints/cae --ae_dir checkpoints/ae --output_dir results/visualization --num_samples 5 --num_vis_samples 5
```

## License :page_facing_up:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact :email:
**Lingfeng Chen**  
:office: National University of Defense Technology  
:e-mail: [chenlingfeng@nudt.edu.cn](mailto:chenlingfeng@nudt.edu.cn)  
:globe_with_meridians: [Personal Homepage](http://lingfengchen.cn/)  
