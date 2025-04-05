## 🌟 Overview

High-Resolution Range Profiles (HRRPs) are often contaminated with noise during acquisition, which can degrade radar system performance. This project provides a unified framework to train and test different deep learning models for HRRP signal denoising. The framework supports multiple denoising models and allows consistent evaluation across different noise levels (PSNR values).

### 🔧 Supported Models

- **CGAN (Conditional GAN)**: A conditional generative adversarial network with feature extractors for target identity and radial length
