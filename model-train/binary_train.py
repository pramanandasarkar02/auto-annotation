#!/usr/bin/env python3
"""
Complete Binary Classification Training Script for 3D Medical Images
Single file containing all necessary components for training a binary segmentation model.
"""

import os
import json
import logging
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# MONAI imports
from monai.config import print_config
from monai.data import DataLoader, CacheDataset, partition_dataset, pad_list_data_collate
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityd, CropForegroundd, RandCropByPosNegLabeld, ResizeWithPadOrCropd,
    ToTensord, EnsureTyped, SpatialPadd, AsDiscrete,
    RandRotate90d, RandFlipd, RandGaussianNoised, RandAdjustContrastd
)
from monai.utils import set_determinism

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print MONAI config for debugging
print_config()

# ===============================================================================
# MODEL DEFINITION
# ===============================================================================

def create_unet3d(in_channels: int = 1, out_channels: int = 1):
    """
    Create a 3D UNet model for binary medical image segmentation.
    
    Args:
        in_channels: Number of input channels (typically 1 for medical images)
        out_channels: Number of output channels (1 for binary classification)
    
    Returns:
        UNet model configured for binary segmentation
    """
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1
    )

# ===============================================================================
# DATA LOADING FUNCTIONS
# ===============================================================================

def get_data_list(images_dir: str, labels_dir: str) -> List[Dict[str, str]]:
    """
    Create a list of image-label pairs from directory structure.
    
    Args:
        images_dir: Directory containing image series folders
        labels_dir: Directory containing label series folders
    
    Returns:
        List of dictionaries with 'image' and 'label' keys
    """
    data_list = []
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        logger.error(f"Images dir {images_dir} or labels dir {labels_dir} does not exist")
        return data_list
    
    logger.info(f"Scanning directories: {images_dir} and {labels_dir}")
    
    for series_id in os.listdir(images_dir):
        image_series_path = os.path.join(images_dir, series_id)
        label_series_path = os.path.join(labels_dir, series_id)
        
        if not os.path.isdir(image_series_path) or not os.path.isdir(label_series_path):
            continue
        
        logger.info(f"Processing series: {series_id}")
        
        for image_id in os.listdir(image_series_path):
            if not image_id.endswith(('.nii', '.nii.gz')):
                continue
                
            image_path = os.path.join(image_series_path, image_id)
            
            # Get base name without extension for matching
            if image_id.endswith('.nii.gz'):
                base_name = image_id[:-7]  # Remove .nii.gz
            else:
                base_name = image_id[:-4]  # Remove .nii
            
            # Find matching label file
            matches = []
            for label_file in os.listdir(label_series_path):
                if base_name in label_file and label_file.endswith(('.nii', '.nii.gz')):
                    matches.append(label_file)
            
            if not matches:
                logger.warning(f"No label found for {image_id}")
                continue
            
            label_path = os.path.join(label_series_path, matches[0])
            
            # Verify both files exist
            if os.path.exists(image_path) and os.path.exists(label_path):
                data_list.append({"image": image_path, "label": label_path})
                logger.info(f"Added pair: {image_id} -> {matches[0]}")
            else:
                logger.warning(f"Missing file: {image_path} or {label_path}")
    
    logger.info(f"Found {len(data_list)} image-label pairs")
    return data_list

# ===============================================================================
# TRANSFORM DEFINITIONS
# ===============================================================================

def get_train_transforms():
    """Get training transforms pipeline with data augmentation."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(keys="image"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
        
        # Data Augmentation
        RandRotate90d(keys=["image", "label"], prob=0.3),
        RandFlipd(keys=["image", "label"], prob=0.3),
        RandGaussianNoised(keys="image", prob=0.2, std=0.1),
        RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.8, 1.2)),
        
        # Random crop for training
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1, neg=1, num_samples=2,  # Reduced for binary
            image_key="image",
            image_threshold=0,
            allow_smaller=True
        ),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])

def get_val_transforms():
    """Get validation transforms pipeline (no augmentation)."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(keys="image"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])

# ===============================================================================
# DATALOADER CREATION
# ===============================================================================

def create_dataloaders(data_list: List[Dict], batch_size: int = 2, cache_rate: float = 1.0):
    """
    Create training and validation dataloaders.
    
    Args:
        data_list: List of dictionaries with image and label paths
        batch_size: Batch size for training
        cache_rate: Cache rate for CacheDataset (1.0 = cache all)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if not data_list:
        raise ValueError("No data found in provided data list.")
    
    logger.info(f"Creating dataloaders from {len(data_list)} samples")
    
    # Split data into train/validation (80/20)
    train_files, val_files = partition_dataset(
        data=data_list, 
        ratios=[0.8, 0.2], 
        shuffle=True
    )
    
    logger.info(f"Train samples: {len(train_files)}, Val samples: {len(val_files)}")
    
    # Get transforms
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()
    
    # Create datasets with caching
    train_ds = CacheDataset(
        data=train_files, 
        transform=train_transforms, 
        cache_rate=cache_rate, 
        num_workers=4
    )
    val_ds = CacheDataset(
        data=val_files, 
        transform=val_transforms, 
        cache_rate=cache_rate, 
        num_workers=4
    )
    
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Empty dataset after transforms!")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=pad_list_data_collate
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    logger.info(f"Created {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    return train_loader, val_loader

# ===============================================================================
# TRAINING FUNCTION
# ===============================================================================

def train_binary_model(
    images_dir: str, 
    labels_dir: str, 
    output_dir: str = "./models",
    max_epochs: int = 50,
    learning_rate: float = 1e-4,
    batch_size: int = 2,
    device: str = None,
    val_interval: int = 1,
    save_interval: int = 10
):
    """
    Main training function for binary 3D medical image segmentation.
    
    Args:
        images_dir: Directory containing training images
        labels_dir: Directory containing training labels
        output_dir: Directory to save models and logs
        max_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        device: Device to use ('cuda' or 'cpu')
        val_interval: Validation frequency (epochs)
        save_interval: Model save frequency (epochs)
    
    Returns:
        Tuple of (model, best_dice, best_epoch)
    """
    # Setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    logger.info(f"Training configuration:")
    logger.info(f"  - Max epochs: {max_epochs}")
    logger.info(f"  - Learning rate: {learning_rate}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Validation interval: {val_interval}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_binary_model.pth")
    
    # Set deterministic training
    set_determinism(seed=42)
    
    # Load data
    logger.info("Loading data...")
    data_list = get_data_list(images_dir, labels_dir)
    
    if len(data_list) == 0:
        raise ValueError("No data found! Check your directory paths.")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_list=data_list,
        batch_size=batch_size,
        cache_rate=1.0
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_unet3d(in_channels=1, out_channels=1).to(device)
    
    # Loss function for binary segmentation
    loss_function = DiceCELoss(sigmoid=True)  # Binary classification
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    
    # Post-processing transforms
    post_pred = AsDiscrete(threshold=0.5)  # Convert probabilities to binary
    post_label = AsDiscrete(threshold=0.5)  # Ensure labels are binary
    
    # Training tracking
    best_metric = -1
    best_metric_epoch = -1
    train_losses = []
    val_dices = []
    
    logger.info("Starting training...")
    
    # Training loop
    for epoch in range(max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{max_epochs}")
        print(f"{'='*60}")
        
        # Training phase
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if step % 5 == 0:
                logger.info(f"[Train] Step {step}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        epoch_loss /= step
        train_losses.append(epoch_loss)
        
        logger.info(f"Epoch {epoch + 1} average training loss: {epoch_loss:.4f}")
        
        # Validation phase
        if (epoch + 1) % val_interval == 0:
            logger.info("Running validation...")
            model.eval()
            
            with torch.no_grad():
                dice_metric.reset()
                val_step = 0
                
                for val_data in val_loader:
                    val_step += 1
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    
                    # Sliding window inference for better results
                    val_outputs = sliding_window_inference(
                        val_inputs, 
                        roi_size=(96, 96, 96), 
                        sw_batch_size=1, 
                        predictor=model
                    )
                    
                    # Post-process predictions and labels
                    val_outputs = post_pred(val_outputs)
                    val_labels = post_label(val_labels)
                    
                    # Calculate dice
                    dice_metric(val_outputs, val_labels)
                    
                    if val_step % 5 == 0:
                        logger.info(f"[Val] Step {val_step}/{len(val_loader)}")
                
                # Get validation metric
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                val_dices.append(metric)
                
                logger.info(f"[Val] Epoch {epoch + 1} Dice: {metric:.4f}")
                
                # Save best model
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"🎉 New best model! Dice: {best_metric:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'dice': val_dices[-1] if val_dices else 0,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Training complete
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETED!")
    logger.info(f"Best Dice Score: {best_metric:.4f} at epoch {best_metric_epoch}")
    logger.info(f"Best model saved: {model_path}")
    logger.info(f"{'='*60}")
    
    # Save training history
    history = {
        "train_losses": train_losses,
        "val_dices": val_dices,
        "best_dice": best_metric,
        "best_epoch": best_metric_epoch,
        "config": {
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "device": str(device)
        }
    }
    
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    logger.info(f"Training history saved: {history_path}")
    
    return model, best_metric, best_metric_epoch

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

if __name__ == "__main__":
    # Configuration
    ROOT_DIR = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data"
    IMAGES_DIR = os.path.join(ROOT_DIR, "image")
    LABELS_DIR = os.path.join(ROOT_DIR, "label")
    OUTPUT_DIR = "./models/binary_segmentation"
    
    # Training parameters
    MAX_EPOCHS = 100
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 2
    
    try:
        # Run training
        model, best_dice, best_epoch = train_binary_model(
            images_dir=IMAGES_DIR,
            labels_dir=LABELS_DIR,
            output_dir=OUTPUT_DIR,
            max_epochs=MAX_EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            val_interval=1,
            save_interval=10
        )
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Best model achieved Dice score of {best_dice:.4f} at epoch {best_epoch}")
        print(f"💾 Model saved in: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise