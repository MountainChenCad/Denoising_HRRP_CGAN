# ablation_cgan.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
import json
from pathlib import Path

# Import models
from models.modules import TargetRadialLengthModule, TargetIdentityModule
from models.cgan_models import Generator, Discriminator
from utils.hrrp_dataset import HRRPDataset
from utils.noise_utils import add_noise_for_psnr, calculate_psnr, calculate_ssim


def train_feature_extractors(args, device):
    """
    Train the Target Radial Length Module (G_D) and Target Identity Module (G_I)

    Args:
        args: Training arguments
        device: Device to train on (CPU or GPU)

    Returns:
        Trained G_D and G_I models
    """
    print(f"Training feature extractors...")

    # Create G_D model
    G_D = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim).to(device)

    # Define loss function and optimizer for G_D
    criterion_GD = nn.MSELoss()
    optimizer_GD = optim.Adam(G_D.parameters(), lr=args.lr)

    # Create G_I model
    G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                               num_classes=args.num_classes).to(device)

    # Define loss function and optimizer for G_I
    criterion_GI = nn.CrossEntropyLoss()
    optimizer_GI = optim.Adam(G_I.parameters(), lr=args.lr)

    # Load dataset
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Update num_classes based on dataset
    num_classes = train_dataset.get_num_classes()
    if num_classes != args.num_classes:
        print(f"Updating num_classes from {args.num_classes} to {num_classes} based on dataset")
        args.num_classes = num_classes
        G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                                   num_classes=args.num_classes).to(device)
        optimizer_GI = optim.Adam(G_I.parameters(), lr=args.lr)

    # Print some info about the dataset
    print(f"Training feature extractors with dataset from: {args.train_dir}")
    print(f"Number of samples: {len(train_dataset)}")
    print(f"Number of classes: {args.num_classes}")

    # Training loop
    for epoch in range(args.feature_epochs):
        epoch_gd_loss = 0.0
        epoch_gi_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for this epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.feature_epochs}")

        for data, radial_length, identity_labels in progress_bar:
            # Move data to device
            data = data.float().to(device)
            radial_length = radial_length.float().to(device)
            identity_labels = identity_labels.long().to(device)

            # ======== Train G_D ========
            optimizer_GD.zero_grad()
            _, predicted_radial_length = G_D(data)

            # Skip samples with invalid radial length
            valid_indices = ~torch.isnan(radial_length) & ~torch.isinf(radial_length) & (radial_length < 1e6)
            if valid_indices.sum() > 0:
                # Use only valid values to compute loss
                gd_loss = criterion_GD(
                    predicted_radial_length[valid_indices],
                    radial_length[valid_indices]
                )

                if not torch.isnan(gd_loss) and not torch.isinf(gd_loss) and gd_loss < 1e6:
                    gd_loss.backward()
                    optimizer_GD.step()
                    epoch_gd_loss += gd_loss.item()

            # ======== Train G_I ========
            optimizer_GI.zero_grad()
            _, identity_logits = G_I(data)
            gi_loss = criterion_GI(identity_logits, identity_labels)
            gi_loss.backward()
            optimizer_GI.step()
            epoch_gi_loss += gi_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(identity_logits.data, 1)
            total += identity_labels.size(0)
            correct += (predicted == identity_labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'G_D Loss': f"{gd_loss.item():.4f}" if 'gd_loss' in locals() else "N/A",
                'G_I Loss': f"{gi_loss.item():.4f}",
                'G_I Acc': f"{100 * correct / total:.2f}%"
            })

        # Calculate epoch metrics
        epoch_gd_loss /= len(train_loader)
        epoch_gi_loss /= len(train_loader)
        epoch_gi_accuracy = 100 * correct / total

        print(f"Epoch {epoch + 1}/{args.feature_epochs} - "
              f"G_D Loss: {epoch_gd_loss:.4f}, "
              f"G_I Loss: {epoch_gi_loss:.4f}, "
              f"G_I Accuracy: {epoch_gi_accuracy:.2f}%")

    return G_D, G_I


def validate_models(generator, G_D, G_I, val_loader, psnr_level, device):
    """
    Validate models on validation data and calculate metrics

    Args:
        generator: Generator model
        G_D: Target Radial Length Module
        G_I: Target Identity Module
        val_loader: Validation data loader
        psnr_level: PSNR level for noise generation
        device: Device to run validation on

    Returns:
        Dictionary of validation metrics
    """
    generator.eval()
    G_D.eval()
    G_I.eval()

    total_psnr = 0
    total_ssim = 0
    total_mse = 0
    n_samples = 0
    mse_loss = nn.MSELoss()

    with torch.no_grad():
        for clean_data, _, _ in tqdm(val_loader, desc="Validation"):
            n_samples += 1
            clean_data = clean_data.float().to(device)

            # Create noisy data
            noisy_data = add_noise_for_psnr(clean_data, psnr_level)

            # Extract features
            f_D, _ = G_D(clean_data)
            f_I, _ = G_I(clean_data)
            condition = torch.cat([f_D, f_I], dim=1)

            # Generate denoised data
            denoised_data = generator(noisy_data, condition)

            # Calculate metrics
            mse = mse_loss(denoised_data, clean_data).item()
            psnr = calculate_psnr(clean_data, denoised_data)

            # Calculate SSIM
            clean_np = clean_data.cpu().numpy()[0]
            denoised_np = denoised_data.cpu().numpy()[0]
            ssim = calculate_ssim(clean_np, denoised_np)

            total_mse += mse
            total_psnr += psnr
            total_ssim += ssim

    # Calculate averages
    avg_mse = total_mse / n_samples
    avg_psnr = total_psnr / n_samples
    avg_ssim = total_ssim / n_samples

    generator.train()

    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'mse': avg_mse
    }


def run_ablation_experiment(args, device, config):
    """
    Run a single ablation experiment with specific configuration

    Args:
        args: Training arguments
        device: Device to train on (CPU or GPU)
        config: Dict containing experiment configuration

    Returns:
        Dict containing results and metrics
    """
    # Extract configuration parameters
    lambda_rec = config.get('lambda_rec', 0.0)  # 0.0 to disable reconstruction loss
    update_g_i = config.get('update_g_i', False)
    update_g_d = config.get('update_g_d', False)
    psnr_level = config.get('psnr_level', 10.0)

    # Create output directory
    config_str = f"rec_{lambda_rec}_gi_{int(update_g_i)}_gd_{int(update_g_d)}"
    output_dir = os.path.join(args.output_dir, f"ablation_{config_str}_psnr_{psnr_level}")
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\n{'=' * 50}")
    print(f"Running experiment: {config_str} at PSNR={psnr_level}dB")
    print(f"lambda_rec={lambda_rec}, update_G_I={update_g_i}, update_G_D={update_g_d}")
    print(f"{'=' * 50}\n")

    # Load or train feature extractors
    if args.feature_extractors_dir and os.path.exists(args.feature_extractors_dir):
        print(f"Loading pre-trained feature extractors from {args.feature_extractors_dir}")
        G_D = TargetRadialLengthModule(input_dim=args.input_dim, feature_dim=args.feature_dim).to(device)
        G_I = TargetIdentityModule(input_dim=args.input_dim, feature_dim=args.feature_dim,
                                   num_classes=args.num_classes).to(device)

        G_D.load_state_dict(torch.load(os.path.join(args.feature_extractors_dir, 'G_D_final.pth')))
        G_I.load_state_dict(torch.load(os.path.join(args.feature_extractors_dir, 'G_I_final.pth')))
    else:
        print("Training feature extractors from scratch...")
        G_D, G_I = train_feature_extractors(args, device)

        # Save feature extractors
        torch.save(G_D.state_dict(), os.path.join(output_dir, 'G_D_initial.pth'))
        torch.save(G_I.state_dict(), os.path.join(output_dir, 'G_I_initial.pth'))

    # Create CGAN models
    generator = Generator(input_dim=args.input_dim,
                          condition_dim=args.feature_dim * 2,
                          hidden_dim=args.hidden_dim).to(device)

    discriminator = Discriminator(input_dim=args.input_dim,
                                  condition_dim=args.feature_dim * 2,
                                  hidden_dim=args.hidden_dim).to(device)

    # Define loss functions
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.MSELoss()
    regression_loss = nn.MSELoss()  # for G_D
    classification_loss = nn.CrossEntropyLoss()  # for G_I

    # Define optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Only create optimizers for G_D and G_I if we're updating them
    if update_g_d:
        optimizer_GD = optim.Adam(G_D.parameters(), lr=args.lr_feature_extractors, betas=(0.5, 0.999))

    if update_g_i:
        optimizer_GI = optim.Adam(G_I.parameters(), lr=args.lr_feature_extractors, betas=(0.5, 0.999))

    # Load dataset
    train_dataset = HRRPDataset(args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Create validation dataset from a subset of training data
    val_size = min(len(train_dataset) // 5, 100)  # 20% or max 100 samples
    val_indices = random.sample(range(len(train_dataset)), val_size)
    train_indices = [i for i in range(len(train_dataset)) if i not in val_indices]

    from torch.utils.data import Subset
    val_dataset = Subset(train_dataset, val_indices)
    train_dataset = Subset(train_dataset, train_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Update num_classes based on dataset (if needed)
    num_classes = train_dataset.dataset.get_num_classes()
    if num_classes != args.num_classes:
        args.num_classes = num_classes

    # Training statistics
    d_losses = []
    g_adv_losses = []
    g_rec_losses = []
    gd_losses = []
    gi_losses = []

    val_psnr_history = []
    val_ssim_history = []
    val_mse_history = []

    best_val_psnr = 0
    best_epoch = 0

    # Training loop
    for epoch in range(args.epochs):
        epoch_d_loss = 0.0
        epoch_g_adv_loss = 0.0
        epoch_g_rec_loss = 0.0
        epoch_gd_loss = 0.0
        epoch_gi_loss = 0.0

        # Create progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        generator.train()
        discriminator.train()
        if update_g_d:
            G_D.train()
        else:
            G_D.eval()

        if update_g_i:
            G_I.train()
        else:
            G_I.eval()

        for i, (data_batch, radial_length, identity_labels) in enumerate(progress_bar):
            # Move data to device
            clean_data = data_batch[0].float().to(device) if isinstance(data_batch,
                                                                        (list, tuple)) else data_batch.float().to(
                device)
            if isinstance(radial_length, (list, tuple)):
                radial_length = radial_length[0].float().to(device)
            else:
                radial_length = radial_length.float().to(device)

            if isinstance(identity_labels, (list, tuple)):
                identity_labels = identity_labels[0].long().to(device)
            else:
                identity_labels = identity_labels.long().to(device)

            batch_size = clean_data.shape[0]

            # Create noisy data at the target PSNR
            noisy_data = add_noise_for_psnr(clean_data, psnr_level)

            # ========================
            # 1. Extract features
            # ========================
            with torch.no_grad():
                f_D, _ = G_D(clean_data)
                f_I, _ = G_I(clean_data)
                condition = torch.cat([f_D, f_I], dim=1)

            # ========================
            # 2. Train Discriminator
            # ========================
            for _ in range(args.n_critic):
                optimizer_D.zero_grad()

                # Generate fake samples
                with torch.no_grad():
                    generated_samples = generator(noisy_data, condition)

                # Create labels with smoothing
                real_labels = torch.ones(batch_size, 1).to(device) * 0.9
                fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1

                # Discriminator loss for real samples
                real_pred = discriminator(clean_data, condition)
                real_loss = adversarial_loss(real_pred, real_labels)

                # Discriminator loss for fake samples
                fake_pred = discriminator(generated_samples.detach(), condition)
                fake_loss = adversarial_loss(fake_pred, fake_labels)

                # Total discriminator loss
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()

                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.clip_value)
                optimizer_D.step()

            # ========================
            # 3. Train Generator
            # ========================
            optimizer_G.zero_grad()

            # Generate samples
            generated_samples = generator(noisy_data, condition)

            # Adversarial loss (fool the discriminator)
            g_adv_loss = adversarial_loss(discriminator(generated_samples, condition), real_labels)

            # Reconstruction loss - may be disabled
            if lambda_rec > 0:
                g_rec_loss = reconstruction_loss(generated_samples, clean_data)
                g_loss = g_adv_loss + lambda_rec * g_rec_loss
            else:
                g_rec_loss = torch.tensor(0.0, device=device)
                g_loss = g_adv_loss

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_value)
            optimizer_G.step()

            # ========================
            # 4. Update G_D (optional)
            # ========================
            if update_g_d:
                optimizer_GD.zero_grad()

                # Predict radial length
                _, pred_radial = G_D(clean_data)

                # Skip samples with invalid radial length
                valid_indices = ~torch.isnan(radial_length) & ~torch.isinf(radial_length) & (radial_length < 1e6)
                if valid_indices.sum() > 0:
                    # Use only valid values to compute loss
                    gd_loss = regression_loss(
                        pred_radial[valid_indices],
                        radial_length[valid_indices]
                    )

                    if not torch.isnan(gd_loss) and not torch.isinf(gd_loss) and gd_loss < 1e6:
                        (args.lambda_gd * gd_loss).backward()
                        torch.nn.utils.clip_grad_norm_(G_D.parameters(), args.clip_value)
                        optimizer_GD.step()
                        epoch_gd_loss += gd_loss.item()

            # ========================
            # 5. Update G_I (optional)
            # ========================
            if update_g_i:
                optimizer_GI.zero_grad()

                # Predict identity
                _, pred_identity = G_I(clean_data)

                # Calculate G_I classification loss
                gi_loss = classification_loss(pred_identity, identity_labels)

                # Apply loss weight and backpropagate
                (args.lambda_gi * gi_loss).backward()
                torch.nn.utils.clip_grad_norm_(G_I.parameters(), args.clip_value)
                optimizer_GI.step()

                epoch_gi_loss += gi_loss.item()

            # Update epoch losses
            epoch_d_loss += d_loss.item()
            epoch_g_adv_loss += g_adv_loss.item()
            epoch_g_rec_loss += g_rec_loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'D_Loss': f"{d_loss.item():.4f}",
                'G_Adv': f"{g_adv_loss.item():.4f}",
                'G_Rec': f"{g_rec_loss.item():.4f}"
            })

        # Calculate average losses for the epoch
        epoch_d_loss /= len(progress_bar)
        epoch_g_adv_loss /= len(progress_bar)
        epoch_g_rec_loss /= len(progress_bar)
        epoch_gd_loss /= max(1, len(progress_bar) if update_g_d else 1)
        epoch_gi_loss /= max(1, len(progress_bar) if update_g_i else 1)

        # Save losses for plotting
        d_losses.append(epoch_d_loss)
        g_adv_losses.append(epoch_g_adv_loss)
        g_rec_losses.append(epoch_g_rec_loss)
        gd_losses.append(epoch_gd_loss)
        gi_losses.append(epoch_gi_loss)

        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"D: {epoch_d_loss:.4f}, G_adv: {epoch_g_adv_loss:.4f}, G_rec: {epoch_g_rec_loss:.4f}, "
              f"G_D: {epoch_gd_loss:.4f}, G_I: {epoch_gi_loss:.4f}")

        # Validate model every few epochs
        if (epoch + 1) % args.val_interval == 0:
            print("Validating model...")
            val_metrics = validate_models(generator, G_D, G_I, val_loader, psnr_level, device)
            val_psnr_history.append(val_metrics['psnr'])
            val_ssim_history.append(val_metrics['ssim'])
            val_mse_history.append(val_metrics['mse'])

            print(f"Validation results - PSNR: {val_metrics['psnr']:.2f}dB, "
                  f"SSIM: {val_metrics['ssim']:.4f}, MSE: {val_metrics['mse']:.6f}")

            # Save best model
            if val_metrics['psnr'] > best_val_psnr:
                best_val_psnr = val_metrics['psnr']
                best_epoch = epoch + 1

                # Save models
                torch.save(generator.state_dict(), os.path.join(output_dir, 'generator_best.pth'))
                torch.save(discriminator.state_dict(), os.path.join(output_dir, 'discriminator_best.pth'))

                if update_g_d:
                    torch.save(G_D.state_dict(), os.path.join(output_dir, 'G_D_best.pth'))

                if update_g_i:
                    torch.save(G_I.state_dict(), os.path.join(output_dir, 'G_I_best.pth'))

        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save models
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, 'generator.pth'))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, 'discriminator.pth'))

            if update_g_d:
                torch.save(G_D.state_dict(), os.path.join(checkpoint_dir, 'G_D.pth'))

            if update_g_i:
                torch.save(G_I.state_dict(), os.path.join(checkpoint_dir, 'G_I.pth'))

    # Save final models
    torch.save(generator.state_dict(), os.path.join(output_dir, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(output_dir, 'discriminator_final.pth'))
    torch.save(G_D.state_dict(), os.path.join(output_dir, 'G_D_final.pth'))
    torch.save(G_I.state_dict(), os.path.join(output_dir, 'G_I_final.pth'))

    # Plot loss curves
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(g_adv_losses, label='Generator Adversarial Loss')
    plt.plot(g_rec_losses, label='Generator Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Losses')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(gd_losses, label='G_D Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('G_D Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(gi_losses, label='G_I Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('G_I Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

    # Plot validation metrics
    if val_psnr_history:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(list(range(args.val_interval, args.epochs + 1, args.val_interval)), val_psnr_history)
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.title('Validation PSNR')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(list(range(args.val_interval, args.epochs + 1, args.val_interval)), val_ssim_history)
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('Validation SSIM')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(list(range(args.val_interval, args.epochs + 1, args.val_interval)), val_mse_history)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Validation MSE')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'validation_metrics.png'))
        plt.close()

    # Final validation
    print("Running final validation...")
    val_metrics = validate_models(generator, G_D, G_I, val_loader, psnr_level, device)

    # Save results summary
    results = {
        'config': config,
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'd_loss': d_losses,
            'g_adv_loss': g_adv_losses,
            'g_rec_loss': g_rec_losses,
            'gd_loss': gd_losses,
            'gi_loss': gi_losses
        },
        'validation': {
            'psnr_history': val_psnr_history,
            'ssim_history': val_ssim_history,
            'mse_history': val_mse_history,
            'best_psnr': best_val_psnr,
            'best_epoch': best_epoch,
            'final_psnr': val_metrics['psnr'],
            'final_ssim': val_metrics['ssim'],
            'final_mse': val_metrics['mse']
        }
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Experiment '{config_str}' completed!")
    print(f"Best validation PSNR: {best_val_psnr:.2f}dB at epoch {best_epoch}")
    print(f"Final validation PSNR: {val_metrics['psnr']:.2f}dB, SSIM: {val_metrics['ssim']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='CGAN ablation experiments')

    # General parameters
    parser.add_argument('--train_dir', type=str, default='datasets/simulated_3/train',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--feature_epochs', type=int, default=500,
                        help='Number of epochs for training feature extractors from scratch')
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='Learning rate')
    parser.add_argument('--input_dim', type=int, default=500,
                        help='Dimension of input HRRP sequence')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Epoch interval for saving checkpoints')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Epoch interval for validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--psnr_levels', type=str, default='5',
                        help='PSNR levels to test at (comma-separated values in dB)')

    # Feature extractors parameters
    parser.add_argument('--feature_extractors_dir', type=str, default='',
                        help='Directory containing pre-trained feature extractors')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='Dimension of feature extractors output')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of target identity classes')

    # CGAN specific parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimension of hidden layers for CGAN')
    parser.add_argument('--lr_feature_extractors', type=float, default=0.00001,
                        help='Learning rate for fine-tuning feature extractors')
    parser.add_argument('--lambda_gd', type=float, default=0.0001,
                        help='Weight of G_D regression loss')
    parser.add_argument('--lambda_gi', type=float, default=0.1,
                        help='Weight of G_I classification loss')
    parser.add_argument('--n_critic', type=int, default=1,
                        help='Number of discriminator updates per generator update')
    parser.add_argument('--clip_value', type=float, default=1.0,
                        help='Gradient clipping value')

    args = parser.parse_args()

    # Parse PSNR levels
    psnr_levels = [float(level) for level in args.psnr_levels.split(',')]

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save training configuration
    with open(os.path.join(args.output_dir, 'training_config.json'), 'w') as f:
        config = {k: v for k, v in vars(args).items()}
        json.dump(config, f, indent=4)

    # Define ablation configurations
    reconstruction_values = [1.0, 10.0]  # Weight of reconstruction loss (0 to disable)
    gi_update_values = [False, True]  # Whether to update G_Ipsnr
    gd_update_values = [False, True]  # Whether to update G_D

    # Generate all experiment configurations
    all_configs = []
    for psnr_level in psnr_levels:
        for lambda_rec in reconstruction_values:
            for update_g_i in gi_update_values:
                for update_g_d in gd_update_values:
                    config = {
                        'lambda_rec': lambda_rec,
                        'update_g_i': update_g_i,
                        'update_g_d': update_g_d,
                        'psnr_level': psnr_level
                    }
                    all_configs.append(config)

    # Run all experiments
    results = []
    for config in all_configs:
        result = run_ablation_experiment(args, device, config)
        results.append(result)

    # Save all results
    with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
        json.dump({
            'args': {k: v for k, v in vars(args).items()},
            'configs': all_configs,
            'results': results
        }, f, indent=4)

    print("\nAll ablation experiments completed!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total ablation experiment time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")