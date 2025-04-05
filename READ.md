# HRRP Denoising Framework 🔍

A comprehensive deep learning framework for High-Resolution Range Profile (HRRP) signal denoising, supporting multiple models and PSNR-controlled training and testing.

## 🌟 Overview

High-Resolution Range Profiles (HRRPs) are often contaminated with noise during acquisition, which can degrade radar system performance. This project provides a unified framework to train and test different deep learning models for HRRP signal denoising. The framework supports multiple denoising models and allows consistent evaluation across different noise levels (PSNR values).

### 🔧 Supported Models

- **CGAN (Conditional GAN)**: A conditional generative adversarial network with feature extractors for target identity and radial length
- **CAE (Convolutional AutoEncoder)**: A deep autoencoder using 1D convolutional layers
- **AE (AutoEncoder)**: A traditional fully-connected autoencoder

## 🚀 Features

- **Unified Training Interface**: Train any model with a single command
- **PSNR-controlled Training**: Train models at specific PSNR noise levels
- **Unified Testing Framework**: Evaluate all models with consistent metrics
- **Comprehensive Comparison**: Compare model performance across different noise conditions
- **Publication-quality Visualization**: Generate high-quality visualization of denoising results

## 📁 Repository Structure

```
├── ae_models.py          # AE model definition
├── cae_models.py         # CAE model definition
├── cgan_models.py        # CGAN generator and discriminator definitions
├── models.py             # Feature extractor modules definitions
├── hrrp_dataset.py       # HRRP dataset loading and preprocessing
├── noise_utils.py        # Noise generation and PSNR utility functions
├── metrics.py            # Evaluation metrics calculation
├── visualization.py      # Visualization tools for results
├── train_all.py          # Unified training interface
├── test_all.py           # Unified testing and comparison interface
├── checkpoints/          # Directory for saved models
│   ├── ae/               # AE model checkpoints
│   ├── cae/              # CAE model checkpoints
│   └── cgan/             # CGAN model checkpoints 
└── results/              # Directory for test results
```

## 📋 Requirements

- Python 3.8
- PyTorch 2.2.0
- CUDA 12.5
- NumPy
- Matplotlib
- Scikit-image
- Pandas
- Seaborn

## 🔰 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/hrrp-denoising.git
cd hrrp-denoising
```

2. Install required packages:

```bash
pip install torch==2.2.0 numpy matplotlib scikit-image pandas seaborn
```

## 💡 Usage

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

## 📊 Metrics and Evaluation

The framework evaluates denoising performance using multiple metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better
- **SSIM (Structural Similarity Index)**: Higher is better
- **MSE (Mean Squared Error)**: Lower is better

All metrics are calculated and reported for each model and noise level.

## 🔬 Experimental Results

After training and testing models using this framework, you'll find:

1. **Individual Model Results**: Performance of each model at different PSNR levels
2. **Comparative Analysis**: Side-by-side comparisons of all models
3. **Visualization**: High-quality plots showing original, noisy, and denoised signals
4. **Summary Reports**: Aggregated performance metrics in text and CSV formats

## 📝 Citation

If you use this code for your research, please cite our work:

```
@article{your_reference,
  title={Denoising HRRP with Deep Learning Models: A Comparative Study},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE.txt file for details.

## Acknowledgments

- This implementation builds upon existing work in conditional GANs and autoencoder architectures
- The framework design is inspired by best practices in deep learning model comparison and evaluation