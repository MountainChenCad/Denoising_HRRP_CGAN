# test_all.py
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from pathlib import Path
import copy
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from models.modules import TargetRadialLengthModule, TargetIdentityModule
from models.cgan_models import Generator
from models.cae_models import ConvAutoEncoder
from models.msae_models import ModifiedSparseAutoEncoder
from utils.hrrp_dataset import HRRPDataset
from torch.utils.data import DataLoader, TensorDataset
from utils.noise_utils import add_noise_for_exact_psnr, calculate_psnr, calculate_ssim
from utils.cnn_evaluator import HRRPCNN, train_cnn, evaluate_cnn, evaluate_denoising_with_cnn

# Set matplotlib parameters for better visualizations
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 13

# Define color scheme for visualizations
COLORS = {
    'noisy': '#7F7F7F',  # Gray
    'cgan': '#40557E',  # Blue
    'cae': '#C9717A',  # Green
    'msae': '#B5B6B6',  # Red
    'clean': '#000000'  # Black
}


def load_or_train_cnn(args, device):
    """
    Load a pre-trained CNN model or train a new one for HRRP classification

    Args:
        args: Arguments containing parameters
        device: Device to use (CPU or GPU)

    Returns:
        Trained CNN model
    """
    model = HRRPCNN(input_dim=args.input_dim, num_classes=args.num_classes).to(device)

    # Check if pre-trained model exists
    cnn_model_path = os.path.join(args.cnn_dir, 'cnn_classifier.pth')

    if os.path.exists(cnn_model_path) and not args.retrain_cnn:
        print(f"Loading pre-trained CNN classifier from {cnn_model_path}")
        model.load_state_dict(torch.load(cnn_model_path, map_location=device))
        return model

    # Train a new model if needed
    print("Training a CNN classifier for evaluation...")

    # Create output directory for CNN model
    os.makedirs(args.cnn_dir, exist_ok=True)

    # Load training dataset
    train_dataset = HRRPDataset(args.train_dir, dataset_type=args.dataset_type)

    # Prepare validation set (20% of training data)
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    # Train CNN model
    train_cnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.cnn_epochs,
        lr=args.cnn_lr,
        device=device,
        save_path=cnn_model_path
    )

    return model


def test_cgan(args, device, psnr_level, cnn_model=None, collected_samples=None):
    """
    Test CGAN model for HRRP signal denoising at a specific PSNR level

    Args:
        args: Testing arguments
        device: Device to test on (CPU or GPU)
        psnr_level: Target PSNR level in dB
        cnn_model: Pre-trained CNN for recognition accuracy evaluation (optional)
        collected_samples: Dictionary to collect samples for grid visualization

    Returns:
        Dictionary of test metrics
    """
    print(f"Testing CGAN for PSNR level {psnr_level}dB...")

    # Create output directory
    output_dir = os.path.join(args.output_dir, f"cgan_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # Determine feature dimensions based on dataset type
    condition_dim = args.feature_dim
    if args.dataset_type == 'simulated':
        condition_dim = args.feature_dim * 2  # Both G_D and G_I features
    else:
        condition_dim = args.feature_dim  # Only G_I features for measured data

    # Load feature extractors
    G_D = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                                   dataset_type=args.dataset_type).to(device)
    G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                               num_classes=args.num_classes).to(device)

    # Load generator
    generator = Generator(input_dim=args.input_dim,
                          condition_dim=condition_dim,
                          hidden_dim=args.hidden_dim,
                          dataset_type=args.dataset_type).to(device)

    # Load model weights
    cgan_dir = os.path.join(args.load_dir, f"cgan_psnr_{psnr_level}dB")

    # Check if model exists
    if not os.path.exists(cgan_dir):
        print(f"No CGAN model found for PSNR={psnr_level}dB at {cgan_dir}")
        return None

    # Load model weights
    G_D.load_state_dict(torch.load(os.path.join(cgan_dir, 'G_D_final.pth'), map_location=device))
    G_I.load_state_dict(torch.load(os.path.join(cgan_dir, 'G_I_final.pth'), map_location=device))
    generator.load_state_dict(torch.load(os.path.join(cgan_dir, 'generator_final.pth'), map_location=device))

    # Set models to evaluation mode
    G_D.eval()
    G_I.eval()
    generator.eval()

    # Load test dataset
    test_dataset = HRRPDataset(args.test_dir, dataset_type=args.dataset_type)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluation metrics
    mse_loss = nn.MSELoss()
    total_noisy_mse = 0
    total_denoised_mse = 0
    total_noisy_psnr = 0
    total_denoised_psnr = 0
    total_denoised_ssim = 0

    # For CNN evaluation
    if cnn_model is not None:
        all_clean_data = []
        all_noisy_data = []
        all_denoised_data = []
        all_labels = []

    # Test samples with progress bar
    progress_bar = tqdm(range(min(args.num_samples, len(test_loader))), desc=f"Testing CGAN PSNR={psnr_level}dB")

    results = []

    with torch.no_grad():
        for i, (clean_data, radial_length, identity_label) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # Move data to device
            clean_data = clean_data.float().to(device)
            identity_label = identity_label.long().to(device)

            # Create noisy data at the target PSNR
            noisy_data, actual_psnr = add_noise_for_exact_psnr(clean_data, psnr_level)

            # Extract features
            f_D, _ = G_D(clean_data)
            f_I, _ = G_I(clean_data)

            # Combine features based on dataset type
            if args.dataset_type == 'simulated':
                condition = torch.cat([f_D, f_I], dim=1)
            else:
                condition = f_I  # For measured data, only use identity features

            # Generate denoised data
            denoised_data = generator(noisy_data, condition)

            # Calculate metrics
            noisy_mse = mse_loss(noisy_data, clean_data).item()
            denoised_mse = mse_loss(denoised_data, clean_data).item()

            noisy_psnr = calculate_psnr(clean_data, noisy_data)
            denoised_psnr = calculate_psnr(clean_data, denoised_data)

            # Calculate SSIM (convert tensors to numpy arrays)
            clean_np = clean_data.cpu().numpy()[0]
            denoised_np = denoised_data.cpu().numpy()[0]
            denoised_ssim = calculate_ssim(clean_np, denoised_np)

            # Accumulate metrics
            total_noisy_mse += noisy_mse
            total_denoised_mse += denoised_mse
            total_noisy_psnr += noisy_psnr
            total_denoised_psnr += denoised_psnr
            total_denoised_ssim += denoised_ssim

            # Store data for CNN evaluation
            if cnn_model is not None:
                all_clean_data.append(clean_data.cpu())
                all_noisy_data.append(noisy_data.cpu())
                all_denoised_data.append(denoised_data.cpu())
                all_labels.append(identity_label.cpu())

            # Store individual results
            results.append({
                'sample_idx': i,
                'noisy_mse': noisy_mse,
                'denoised_mse': denoised_mse,
                'noisy_psnr': noisy_psnr,
                'denoised_psnr': denoised_psnr,
                'denoised_ssim': denoised_ssim,
                'psnr_improvement': denoised_psnr - noisy_psnr
            })

            # Collect samples for grid visualization (first 5 samples)
            if collected_samples is not None and i < 5:
                if i not in collected_samples:
                    collected_samples[i] = {
                        'clean': clean_np,
                        'noisy': noisy_data.cpu().numpy()[0],
                        'noisy_psnr': noisy_psnr
                    }

                # Add CGAN results
                collected_samples[i]['cgan'] = denoised_np
                collected_samples[i]['cgan_psnr'] = denoised_psnr
                collected_samples[i]['cgan_ssim'] = denoised_ssim

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'Noisy PSNR': f"{noisy_psnr:.2f}dB",
                'Denoised PSNR': f"{denoised_psnr:.2f}dB",
                'Improvement': f"{denoised_psnr - noisy_psnr:.2f}dB"
            })

    # Calculate average metrics
    n_samples = min(args.num_samples, len(test_loader))
    avg_noisy_mse = total_noisy_mse / n_samples
    avg_denoised_mse = total_denoised_mse / n_samples
    avg_noisy_psnr = total_noisy_psnr / n_samples
    avg_denoised_psnr = total_denoised_psnr / n_samples
    avg_denoised_ssim = total_denoised_ssim / n_samples
    avg_psnr_improvement = avg_denoised_psnr - avg_noisy_psnr

    # CNN-based evaluation if model is provided
    cnn_metrics = None
    if cnn_model is not None and all_clean_data:
        # Combine all batches
        all_clean_data = torch.cat(all_clean_data, dim=0)
        all_noisy_data = torch.cat(all_noisy_data, dim=0)
        all_denoised_data = torch.cat(all_denoised_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Create dataloaders for evaluation
        clean_loader = DataLoader(TensorDataset(all_clean_data, all_labels), batch_size=args.batch_size)
        noisy_loader = DataLoader(TensorDataset(all_noisy_data, all_labels), batch_size=args.batch_size)
        denoised_loader = DataLoader(TensorDataset(all_denoised_data, all_labels), batch_size=args.batch_size)

        # Evaluate recognition accuracy
        print("Evaluating recognition accuracy with CNN...")
        _, clean_acc = evaluate_cnn(cnn_model, clean_loader, device=device)
        _, noisy_acc = evaluate_cnn(cnn_model, noisy_loader, device=device)
        _, denoised_acc = evaluate_cnn(cnn_model, denoised_loader, device=device)

        cnn_metrics = {
            'clean_accuracy': clean_acc,
            'noisy_accuracy': noisy_acc,
            'denoised_accuracy': denoised_acc,
            'accuracy_improvement': denoised_acc - noisy_acc
        }

    # Save summary metrics
    summary = {
        'model': 'CGAN',
        'psnr_level': psnr_level,
        'avg_noisy_mse': avg_noisy_mse,
        'avg_denoised_mse': avg_denoised_mse,
        'avg_noisy_psnr': avg_noisy_psnr,
        'avg_denoised_psnr': avg_denoised_psnr,
        'avg_denoised_ssim': avg_denoised_ssim,
        'avg_psnr_improvement': avg_psnr_improvement,
        'individual_results': results
    }

    # Add CNN metrics if available
    if cnn_metrics:
        summary.update(cnn_metrics)

    # Print summary
    print(f"\nCGAN Results for PSNR={psnr_level}dB:")
    print(f"  Average Noisy PSNR: {avg_noisy_psnr:.2f}dB")
    print(f"  Average Denoised PSNR: {avg_denoised_psnr:.2f}dB")
    print(f"  Average PSNR Improvement: {avg_psnr_improvement:.2f}dB")
    print(f"  Average Denoised SSIM: {avg_denoised_ssim:.4f}")

    return summary


def test_cae(args, device, psnr_level, cnn_model=None, collected_samples=None):
    """
    Test CAE model for HRRP signal denoising at a specific PSNR level

    Args:
        args: Testing arguments
        device: Device to test on (CPU or GPU)
        psnr_level: Target PSNR level in dB
        cnn_model: Pre-trained CNN for recognition accuracy evaluation (optional)
        collected_samples: Dictionary to collect samples for grid visualization

    Returns:
        Dictionary of test metrics
    """
    print(f"Testing CAE for PSNR level {psnr_level}dB...")

    # Create output directory
    output_dir = os.path.join(args.output_dir, f"cae_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # Load CAE model
    model = ConvAutoEncoder(input_dim=args.input_dim,
                            latent_dim=args.latent_dim,
                            hidden_dim=args.hidden_dim).to(device)

    # Load model weights
    cae_dir = os.path.join(args.load_dir, f"cae_psnr_{psnr_level}dB")

    # Check if model exists
    if not os.path.exists(cae_dir):
        print(f"No CAE model found for PSNR={psnr_level}dB at {cae_dir}")
        return None

    # Load model weights
    model.load_state_dict(torch.load(os.path.join(cae_dir, 'cae_model_final.pth'), map_location=device))

    # Set model to evaluation mode
    model.eval()

    # Load test dataset
    test_dataset = HRRPDataset(args.test_dir, dataset_type=args.dataset_type)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluation metrics
    mse_loss = nn.MSELoss()
    total_noisy_mse = 0
    total_denoised_mse = 0
    total_noisy_psnr = 0
    total_denoised_psnr = 0
    total_denoised_ssim = 0

    # For CNN evaluation
    if cnn_model is not None:
        all_clean_data = []
        all_noisy_data = []
        all_denoised_data = []
        all_labels = []

    # Test samples with progress bar
    progress_bar = tqdm(range(min(args.num_samples, len(test_loader))), desc=f"Testing CAE PSNR={psnr_level}dB")

    results = []

    with torch.no_grad():
        for i, (clean_data, _, identity_label) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # Move data to device
            clean_data = clean_data.float().to(device)
            identity_label = identity_label.long().to(device)

            # Create noisy data at the target PSNR
            noisy_data, actual_psnr = add_noise_for_exact_psnr(clean_data, psnr_level)

            # Generate denoised data
            denoised_data, _ = model(noisy_data)

            # Calculate metrics
            noisy_mse = mse_loss(noisy_data, clean_data).item()
            denoised_mse = mse_loss(denoised_data, clean_data).item()

            noisy_psnr = calculate_psnr(clean_data, noisy_data)
            denoised_psnr = calculate_psnr(clean_data, denoised_data)

            # Calculate SSIM (convert tensors to numpy arrays)
            clean_np = clean_data.cpu().numpy()[0]
            denoised_np = denoised_data.cpu().numpy()[0]
            denoised_ssim = calculate_ssim(clean_np, denoised_np)

            # Accumulate metrics
            total_noisy_mse += noisy_mse
            total_denoised_mse += denoised_mse
            total_noisy_psnr += noisy_psnr
            total_denoised_psnr += denoised_psnr
            total_denoised_ssim += denoised_ssim

            # Store data for CNN evaluation
            if cnn_model is not None:
                all_clean_data.append(clean_data.cpu())
                all_noisy_data.append(noisy_data.cpu())
                all_denoised_data.append(denoised_data.cpu())
                all_labels.append(identity_label.cpu())

            # Store individual results
            results.append({
                'sample_idx': i,
                'noisy_mse': noisy_mse,
                'denoised_mse': denoised_mse,
                'noisy_psnr': noisy_psnr,
                'denoised_psnr': denoised_psnr,
                'denoised_ssim': denoised_ssim,
                'psnr_improvement': denoised_psnr - noisy_psnr
            })

            # Collect samples for grid visualization (first 5 samples)
            if collected_samples is not None and i < 5:
                if i not in collected_samples:
                    collected_samples[i] = {
                        'clean': clean_np,
                        'noisy': noisy_data.cpu().numpy()[0],
                        'noisy_psnr': noisy_psnr
                    }

                # Add CAE results
                collected_samples[i]['cae'] = denoised_np
                collected_samples[i]['cae_psnr'] = denoised_psnr
                collected_samples[i]['cae_ssim'] = denoised_ssim

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'Noisy PSNR': f"{noisy_psnr:.2f}dB",
                'Denoised PSNR': f"{denoised_psnr:.2f}dB",
                'Improvement': f"{denoised_psnr - noisy_psnr:.2f}dB"
            })

    # Calculate average metrics
    n_samples = min(args.num_samples, len(test_loader))
    avg_noisy_mse = total_noisy_mse / n_samples
    avg_denoised_mse = total_denoised_mse / n_samples
    avg_noisy_psnr = total_noisy_psnr / n_samples
    avg_denoised_psnr = total_denoised_psnr / n_samples
    avg_denoised_ssim = total_denoised_ssim / n_samples
    avg_psnr_improvement = avg_denoised_psnr - avg_noisy_psnr

    # CNN-based evaluation if model is provided
    cnn_metrics = None
    if cnn_model is not None and all_clean_data:
        # Combine all batches
        all_clean_data = torch.cat(all_clean_data, dim=0)
        all_noisy_data = torch.cat(all_noisy_data, dim=0)
        all_denoised_data = torch.cat(all_denoised_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Create dataloaders for evaluation
        clean_loader = DataLoader(TensorDataset(all_clean_data, all_labels), batch_size=args.batch_size)
        noisy_loader = DataLoader(TensorDataset(all_noisy_data, all_labels), batch_size=args.batch_size)
        denoised_loader = DataLoader(TensorDataset(all_denoised_data, all_labels), batch_size=args.batch_size)

        # Evaluate recognition accuracy
        print("Evaluating recognition accuracy with CNN...")
        _, clean_acc = evaluate_cnn(cnn_model, clean_loader, device=device)
        _, noisy_acc = evaluate_cnn(cnn_model, noisy_loader, device=device)
        _, denoised_acc = evaluate_cnn(cnn_model, denoised_loader, device=device)

        cnn_metrics = {
            'clean_accuracy': clean_acc,
            'noisy_accuracy': noisy_acc,
            'denoised_accuracy': denoised_acc,
            'accuracy_improvement': denoised_acc - noisy_acc
        }

    # Save summary metrics
    summary = {
        'model': 'CAE',
        'psnr_level': psnr_level,
        'avg_noisy_mse': avg_noisy_mse,
        'avg_denoised_mse': avg_denoised_mse,
        'avg_noisy_psnr': avg_noisy_psnr,
        'avg_denoised_psnr': avg_denoised_psnr,
        'avg_denoised_ssim': avg_denoised_ssim,
        'avg_psnr_improvement': avg_psnr_improvement,
        'individual_results': results
    }

    # Add CNN metrics if available
    if cnn_metrics:
        summary.update(cnn_metrics)

    # Print summary
    print(f"\nCAE Results for PSNR={psnr_level}dB:")
    print(f"  Average Noisy PSNR: {avg_noisy_psnr:.2f}dB")
    print(f"  Average Denoised PSNR: {avg_denoised_psnr:.2f}dB")
    print(f"  Average PSNR Improvement: {avg_psnr_improvement:.2f}dB")
    print(f"  Average Denoised SSIM: {avg_denoised_ssim:.4f}")

    return summary


def test_msae(args, device, psnr_level, cnn_model=None, collected_samples=None):
    """
    Test MSAE model for HRRP signal denoising at a specific PSNR level

    Args:
        args: Testing arguments
        device: Device to test on (CPU or GPU)
        psnr_level: Target PSNR level in dB
        cnn_model: Pre-trained CNN for recognition accuracy evaluation (optional)
        collected_samples: Dictionary to collect samples for grid visualization

    Returns:
        Dictionary of test metrics
    """
    print(f"Testing MSAE for PSNR level {psnr_level}dB...")

    # Create output directory
    output_dir = os.path.join(args.output_dir, f"msae_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # Load MSAE model
    model = ModifiedSparseAutoEncoder(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.msae_hidden_dim,
        sparsity_param=args.sparsity_param,
        reg_lambda=args.reg_lambda,
        sparsity_beta=args.sparsity_beta
    ).to(device)

    # Load model weights
    msae_dir = os.path.join(args.load_dir, f"msae_psnr_{psnr_level}dB")

    # Check if model exists
    if not os.path.exists(msae_dir):
        print(f"No MSAE model found for PSNR={psnr_level}dB at {msae_dir}")
        return None

    # Load model weights
    model.load_state_dict(torch.load(os.path.join(msae_dir, 'msae_model_final.pth'), map_location=device))

    # Set model to evaluation mode
    model.eval()

    # Load test dataset
    test_dataset = HRRPDataset(args.test_dir, dataset_type=args.dataset_type)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluation metrics
    mse_loss = nn.MSELoss()
    total_noisy_mse = 0
    total_denoised_mse = 0
    total_noisy_psnr = 0
    total_denoised_psnr = 0
    total_denoised_ssim = 0

    # For CNN evaluation
    if cnn_model is not None:
        all_clean_data = []
        all_noisy_data = []
        all_denoised_data = []
        all_labels = []

    # Test samples with progress bar
    progress_bar = tqdm(range(min(args.num_samples, len(test_loader))), desc=f"Testing MSAE PSNR={psnr_level}dB")

    results = []

    with torch.no_grad():
        for i, (clean_data, _, identity_label) in enumerate(test_loader):
            if i >= args.num_samples:
                break

            # Move data to device
            clean_data = clean_data.float().to(device)
            identity_label = identity_label.long().to(device)

            # Create noisy data at the target PSNR
            noisy_data, actual_psnr = add_noise_for_exact_psnr(clean_data, psnr_level)

            # Generate denoised data
            denoised_data, _ = model(noisy_data)

            # Calculate metrics
            noisy_mse = mse_loss(noisy_data, clean_data).item()
            denoised_mse = mse_loss(denoised_data, clean_data).item()

            noisy_psnr = calculate_psnr(clean_data, noisy_data)
            denoised_psnr = calculate_psnr(clean_data, denoised_data)

            # Calculate SSIM (convert tensors to numpy arrays)
            clean_np = clean_data.cpu().numpy()[0]
            denoised_np = denoised_data.cpu().numpy()[0]
            denoised_ssim = calculate_ssim(clean_np, denoised_np)

            # Accumulate metrics
            total_noisy_mse += noisy_mse
            total_denoised_mse += denoised_mse
            total_noisy_psnr += noisy_psnr
            total_denoised_psnr += denoised_psnr
            total_denoised_ssim += denoised_ssim

            # Store data for CNN evaluation
            if cnn_model is not None:
                all_clean_data.append(clean_data.cpu())
                all_noisy_data.append(noisy_data.cpu())
                all_denoised_data.append(denoised_data.cpu())
                all_labels.append(identity_label.cpu())

            # Store individual results
            results.append({
                'sample_idx': i,
                'noisy_mse': noisy_mse,
                'denoised_mse': denoised_mse,
                'noisy_psnr': noisy_psnr,
                'denoised_psnr': denoised_psnr,
                'denoised_ssim': denoised_ssim,
                'psnr_improvement': denoised_psnr - noisy_psnr
            })

            # Collect samples for grid visualization (first 5 samples)
            if collected_samples is not None and i < 5:
                if i not in collected_samples:
                    collected_samples[i] = {
                        'clean': clean_np,
                        'noisy': noisy_data.cpu().numpy()[0],
                        'noisy_psnr': noisy_psnr
                    }

                # Add MSAE results
                collected_samples[i]['msae'] = denoised_np
                collected_samples[i]['msae_psnr'] = denoised_psnr
                collected_samples[i]['msae_ssim'] = denoised_ssim

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'Noisy PSNR': f"{noisy_psnr:.2f}dB",
                'Denoised PSNR': f"{denoised_psnr:.2f}dB",
                'Improvement': f"{denoised_psnr - noisy_psnr:.2f}dB"
            })

    # Calculate average metrics
    n_samples = min(args.num_samples, len(test_loader))
    avg_noisy_mse = total_noisy_mse / n_samples
    avg_denoised_mse = total_denoised_mse / n_samples
    avg_noisy_psnr = total_noisy_psnr / n_samples
    avg_denoised_psnr = total_denoised_psnr / n_samples
    avg_denoised_ssim = total_denoised_ssim / n_samples
    avg_psnr_improvement = avg_denoised_psnr - avg_noisy_psnr

    # CNN-based evaluation if model is provided
    cnn_metrics = None
    if cnn_model is not None and all_clean_data:
        # Combine all batches
        all_clean_data = torch.cat(all_clean_data, dim=0)
        all_noisy_data = torch.cat(all_noisy_data, dim=0)
        all_denoised_data = torch.cat(all_denoised_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Create dataloaders for evaluation
        clean_loader = DataLoader(TensorDataset(all_clean_data, all_labels), batch_size=args.batch_size)
        noisy_loader = DataLoader(TensorDataset(all_noisy_data, all_labels), batch_size=args.batch_size)
        denoised_loader = DataLoader(TensorDataset(all_denoised_data, all_labels), batch_size=args.batch_size)

        # Evaluate recognition accuracy
        print("Evaluating recognition accuracy with CNN...")
        _, clean_acc = evaluate_cnn(cnn_model, clean_loader, device=device)
        _, noisy_acc = evaluate_cnn(cnn_model, noisy_loader, device=device)
        _, denoised_acc = evaluate_cnn(cnn_model, denoised_loader, device=device)

        cnn_metrics = {
            'clean_accuracy': clean_acc,
            'noisy_accuracy': noisy_acc,
            'denoised_accuracy': denoised_acc,
            'accuracy_improvement': denoised_acc - noisy_acc
        }

    # Save summary metrics
    summary = {
        'model': 'MSAE',
        'psnr_level': psnr_level,
        'avg_noisy_mse': avg_noisy_mse,
        'avg_denoised_mse': avg_denoised_mse,
        'avg_noisy_psnr': avg_noisy_psnr,
        'avg_denoised_psnr': avg_denoised_psnr,
        'avg_denoised_ssim': avg_denoised_ssim,
        'avg_psnr_improvement': avg_psnr_improvement,
        'individual_results': results
    }

    # Add CNN metrics if available
    if cnn_metrics:
        summary.update(cnn_metrics)

    # Print summary
    print(f"\nMSAE Results for PSNR={psnr_level}dB:")
    print(f"  Average Noisy PSNR: {avg_noisy_psnr:.2f}dB")
    print(f"  Average Denoised PSNR: {avg_denoised_psnr:.2f}dB")
    print(f"  Average PSNR Improvement: {avg_psnr_improvement:.2f}dB")
    print(f"  Average Denoised SSIM: {avg_denoised_ssim:.4f}")

    return summary


def create_sample_grid_plot(args, psnr_level, collected_samples):
    """
    Create a 5x1 grid plot showing 5 HRRP samples across all methods

    Args:
        args: Testing arguments
        psnr_level: PSNR level in dB
        collected_samples: Dictionary of collected samples data
    """
    print(f"Creating sample visualization grid for PSNR={psnr_level}dB...")

    # Create output directory
    output_dir = os.path.join(args.output_dir, f"comparison_psnr_{psnr_level}dB")
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have samples to visualize
    if not collected_samples or len(collected_samples) == 0:
        print("No samples collected for visualization")
        return

    # Create figure with 5 rows (samples) and 1 column
    fig = plt.figure(figsize=(15, 12))

    # For each of the 5 samples
    for sample_idx in range(min(5, len(collected_samples))):
        sample = collected_samples[sample_idx]

        # Create grid for this sample with 1 row and 5 columns (clean, noisy, 3 methods)
        gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gridspec.GridSpec(5, 1)[sample_idx])

        # Clean signal plot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(sample['clean'], color=COLORS['clean'], linewidth=1.5)
        if sample_idx == 0:  # Title only for the first row
            ax1.set_title('Clean HRRP')
        ax1.set_ylabel(f'Sample {sample_idx + 1}')
        ax1.grid(True, linestyle='--', alpha=0.3)

        # Noisy signal plot
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(sample['noisy'], color=COLORS['noisy'], linewidth=1.5)
        if sample_idx == 0:
            ax2.set_title(f'Noisy HRRP\n(PSNR: {sample["noisy_psnr"]:.2f}dB)')
        else:
            ax2.set_title(f'PSNR: {sample["noisy_psnr"]:.2f}dB')
        ax2.grid(True, linestyle='--', alpha=0.3)

        # CGAN results, if available
        if 'cgan' in sample:
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(sample['cgan'], color=COLORS['cgan'], linewidth=1.5)
            if sample_idx == 0:
                ax3.set_title(f'CGAN Denoising\n(PSNR: {sample["cgan_psnr"]:.2f}dB)')
            else:
                ax3.set_title(f'PSNR: {sample["cgan_psnr"]:.2f}dB')
            ax3.grid(True, linestyle='--', alpha=0.3)

        # CAE results, if available
        if 'cae' in sample:
            ax4 = fig.add_subplot(gs[3])
            ax4.plot(sample['cae'], color=COLORS['cae'], linewidth=1.5)
            if sample_idx == 0:
                ax4.set_title(f'CAE Denoising\n(PSNR: {sample["cae_psnr"]:.2f}dB)')
            else:
                ax4.set_title(f'PSNR: {sample["cae_psnr"]:.2f}dB')
            ax4.grid(True, linestyle='--', alpha=0.3)

        # MSAE results, if available
        if 'msae' in sample:
            ax5 = fig.add_subplot(gs[4])
            ax5.plot(sample['msae'], color=COLORS['msae'], linewidth=1.5)
            if sample_idx == 0:
                ax5.set_title(f'MSAE Denoising\n(PSNR: {sample["msae_psnr"]:.2f}dB)')
            else:
                ax5.set_title(f'PSNR: {sample["msae_psnr"]:.2f}dB')
            ax5.grid(True, linestyle='--', alpha=0.3)

    # Add a common x-label
    fig.text(0.5, 0.04, 'Range Bin', ha='center', fontsize=12)

    # Add a common y-label
    fig.text(0.04, 0.5, 'Magnitude', va='center', rotation='vertical', fontsize=12)

    # Add a super title
    fig.suptitle(f'HRRP Denoising Comparison at PSNR={psnr_level}dB', fontsize=16, y=0.98)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    # Save figure
    plt.savefig(os.path.join(output_dir, 'sample_grid_comparison.svg'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'sample_grid_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_grid(args, all_results):
    """
    Create a 2x2 grid of bar charts showing metrics across all PSNR levels
    for each method: Recognition rate, Recognition rate improvement, SSIM, and PSNR improvement

    Args:
        args: Testing arguments
        all_results: Dictionary of results from all models and PSNR levels
    """
    print("Creating 2x2 comparison grid across all PSNR levels...")

    # Create output directory
    output_dir = os.path.join(args.output_dir, "comparison_grid")
    os.makedirs(output_dir, exist_ok=True)

    # Extract PSNR levels and methods
    psnr_levels = sorted(list(all_results.keys()))
    methods = sorted(list(all_results[psnr_levels[0]].keys()))

    # Check if CNN metrics are available for all methods and PSNR levels
    has_cnn_metrics = True
    for psnr in psnr_levels:
        for method in methods:
            if 'denoised_accuracy' not in all_results[psnr][method]:
                has_cnn_metrics = False
                break

    if not has_cnn_metrics:
        print("Warning: CNN metrics not available for all models. Cannot create comparison grid.")
        return

    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Set bar width and positions
    num_methods = len(methods)
    num_psnr = len(psnr_levels)
    bar_width = 0.8 / num_methods

    # 1. Recognition Rate (top-left)
    ax1 = fig.add_subplot(gs[0, 0])

    for i, method in enumerate(methods):
        x_positions = np.arange(num_psnr)
        accuracy_values = [all_results[psnr][method]['denoised_accuracy'] for psnr in psnr_levels]

        offset = (i - num_methods / 2 + 0.5) * bar_width
        bars = ax1.bar(x_positions + offset, accuracy_values, bar_width,
                       label=method, color=COLORS[method.lower()], edgecolor='k', linewidth=1)

        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    ax1.set_title('Recognition Accuracy', fontsize=13)
    ax1.set_xlabel('Noise Level (PSNR, dB)', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_xticks(np.arange(num_psnr))
    ax1.set_xticklabels([f'{psnr}dB' for psnr in psnr_levels])
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. Recognition Rate Improvement (top-right)
    ax2 = fig.add_subplot(gs[0, 1])

    for i, method in enumerate(methods):
        x_positions = np.arange(num_psnr)
        improvement_values = [all_results[psnr][method]['accuracy_improvement'] for psnr in psnr_levels]

        offset = (i - num_methods / 2 + 0.5) * bar_width
        bars = ax2.bar(x_positions + offset, improvement_values, bar_width,
                       label=method, color=COLORS[method.lower()], edgecolor='k', linewidth=1)

        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'+{height:.1f}%', ha='center', va='bottom', fontsize=8)

    ax2.set_title('Recognition Accuracy Improvement', fontsize=13)
    ax2.set_xlabel('Noise Level (PSNR, dB)', fontsize=11)
    ax2.set_ylabel('Improvement (%)', fontsize=11)
    ax2.set_xticks(np.arange(num_psnr))
    ax2.set_xticklabels([f'{psnr}dB' for psnr in psnr_levels])
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 3. SSIM (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])

    for i, method in enumerate(methods):
        x_positions = np.arange(num_psnr)
        ssim_values = [all_results[psnr][method]['avg_denoised_ssim'] for psnr in psnr_levels]

        offset = (i - num_methods / 2 + 0.5) * bar_width
        bars = ax3.bar(x_positions + offset, ssim_values, bar_width,
                       label=method, color=COLORS[method.lower()], edgecolor='k', linewidth=1)

        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax3.set_title('Structural Similarity Index (SSIM)', fontsize=13)
    ax3.set_xlabel('Noise Level (PSNR, dB)', fontsize=11)
    ax3.set_ylabel('SSIM', fontsize=11)
    ax3.set_xticks(np.arange(num_psnr))
    ax3.set_xticklabels([f'{psnr}dB' for psnr in psnr_levels])
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Set SSIM axis to better show differences
    ssim_min = min([all_results[psnr][method]['avg_denoised_ssim']
                    for psnr in psnr_levels for method in methods]) * 0.95
    ax3.set_ylim(ssim_min, 1.0)

    # 4. PSNR Improvement (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])

    for i, method in enumerate(methods):
        x_positions = np.arange(num_psnr)
        psnr_improvements = [all_results[psnr][method]['avg_psnr_improvement'] for psnr in psnr_levels]

        offset = (i - num_methods / 2 + 0.5) * bar_width
        bars = ax4.bar(x_positions + offset, psnr_improvements, bar_width,
                       label=method, color=COLORS[method.lower()], edgecolor='k', linewidth=1)

        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'+{height:.1f}dB', ha='center', va='bottom', fontsize=8)

    ax4.set_title('PSNR Improvement', fontsize=13)
    ax4.set_xlabel('Noise Level (PSNR, dB)', fontsize=11)
    ax4.set_ylabel('Improvement (dB)', fontsize=11)
    ax4.set_xticks(np.arange(num_psnr))
    ax4.set_xticklabels([f'{psnr}dB' for psnr in psnr_levels])
    ax4.grid(axis='y', linestyle='--', alpha=0.3)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Add a common legend at the bottom
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=len(methods), frameon=False, fontsize=11)

    # Add a super title
    fig.suptitle('Comparative Performance of HRRP Denoising Methods', fontsize=16, y=0.98)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.94])

    # Save figure
    plt.savefig(os.path.join(output_dir, 'metrics_comparison_grid.svg'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'metrics_comparison_grid.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Metrics comparison grid saved to {output_dir}")


def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Unified testing script for HRRP denoising models')

    # General parameters
    parser.add_argument('--model', type=str, default='all',
                        choices=['cgan', 'cae', 'msae', 'all'],
                        help='Model to test')
    parser.add_argument('--test_dir', type=str, default='datasets/simulated_3/test',
                        help='Directory containing test data')
    parser.add_argument('--train_dir', type=str, default='datasets/simulated_3/train',
                        help='Directory containing training data (for CNN training)')
    parser.add_argument('--load_dir', type=str, default='checkpoints_simulated',
                        help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save test results')
    parser.add_argument('--num_samples', type=int, default=300,
                        help='Number of test samples to process')
    parser.add_argument('--input_dim', type=int, default=500,
                        help='Dimension of input HRRP sequence')
    parser.add_argument('--psnr_levels', type=str, default='20,10,5',
                        help='PSNR levels to test at (comma-separated values in dB)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--dataset_type', type=str, default='simulated',
                        choices=['simulated', 'measured'],
                        help='Type of dataset to use (simulated or measured)')

    # Feature extractors parameters
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Dimension of feature extractors output')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of target identity classes')

    # CGAN specific parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimension of hidden layers for CGAN and CAE')

    # CAE and MSAE specific parameters
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of latent space for CAE and AE')
    parser.add_argument('--msae_hidden_dim', type=int, default=128,
                        help='Dimension of hidden layers for MSAE')
    parser.add_argument('--sparsity_param', type=float, default=0.05,
                        help='Sparsity parameter (p) for MSAE')
    parser.add_argument('--reg_lambda', type=float, default=0.0001,
                        help='Weight regularization parameter (lambda) for MSAE')
    parser.add_argument('--sparsity_beta', type=float, default=3.0,
                        help='Sparsity weight parameter (beta) for MSAE')

    # CNN evaluation parameters - Force enable for metrics visualization
    parser.add_argument('--use_cnn_eval', action='store_true', default=True,
                        help='Use CNN for recognition accuracy evaluation')
    parser.add_argument('--cnn_dir', type=str, default='checkpoints/cnn_classifier',
                        help='Directory to save/load CNN classifier')
    parser.add_argument('--retrain_cnn', default=True,
                        help='Force retraining of CNN classifier even if one exists')
    parser.add_argument('--cnn_epochs', type=int, default=20,
                        help='Number of epochs for CNN training')
    parser.add_argument('--cnn_lr', type=float, default=0.001,
                        help='Learning rate for CNN training')

    args = parser.parse_args()

    # Parse PSNR levels
    psnr_levels = [float(level) for level in args.psnr_levels.split(',')]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset type: {args.dataset_type}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load or train CNN model for evaluation
    print(f"Setting up CNN classifier for evaluation metrics...")
    cnn_model = load_or_train_cnn(args, device)
    print("CNN classifier ready for evaluation")

    # Store all results
    all_results = {}

    # Test models at each PSNR level
    for psnr_level in psnr_levels:
        print(f"\n{'=' * 50}")
        print(f"Testing at PSNR level: {psnr_level}dB")
        print(f"{'=' * 50}\n")

        # Results for this PSNR level
        psnr_results = {}

        # Dictionary to collect first 5 samples for grid visualization
        collected_samples = {}

        if args.model in ['cgan', 'all']:
            cgan_result = test_cgan(args, device, psnr_level, cnn_model, collected_samples)
            if cgan_result:
                psnr_results['CGAN'] = cgan_result

        if args.model in ['cae', 'all']:
            cae_result = test_cae(args, device, psnr_level, cnn_model, collected_samples)
            if cae_result:
                psnr_results['CAE'] = cae_result

        if args.model in ['msae', 'all']:
            msae_result = test_msae(args, device, psnr_level, cnn_model, collected_samples)
            if msae_result:
                psnr_results['MSAE'] = msae_result

        # Create sample grid visualization
        if collected_samples and len(collected_samples) > 0:
            create_sample_grid_plot(args, psnr_level, collected_samples)

        # Store results for this PSNR level
        all_results[psnr_level] = psnr_results

    # Create 2x2 grid comparison visualization across all PSNR levels
    if all_results and len(all_results) > 0:
        create_comparison_grid(args, all_results)

    print("\nTesting complete for all models and PSNR levels.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total testing time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")