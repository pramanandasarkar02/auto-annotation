import os
import torch
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, ToTensord, EnsureTyped
)
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import Dataset, DataLoader
from data_loading import get_data_list, create_dataloaders
from model import create_qct_segmentation_model

def create_inference_transforms():
    """Create transforms for inference (similar to validation transforms)"""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"])
    ])

def save_nifti_prediction(prediction, reference_path, output_path):
    """
    Save prediction as NIfTI file using reference image for affine and header
    
    Args:
        prediction: numpy array of prediction
        reference_path: path to reference image for affine/header info
        output_path: path to save the prediction
    """
    # Load reference image to get affine and header
    ref_img = nib.load(reference_path)
    
    # Create new NIfTI image with prediction data
    pred_img = nib.Nifti1Image(prediction.astype(np.float32), ref_img.affine, ref_img.header)
    
    # Save the prediction
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(pred_img, output_path)
    print(f"Prediction saved to: {output_path}")

def calculate_metrics(pred_tensor, label_tensor, num_classes=4):
    """
    Calculate segmentation metrics for multiclass segmentation
    """
    # Initialize metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd_metric = HausdorffDistanceMetric(include_background=True, reduction="mean_batch")
    
    # Ensure tensors are in the right format for metrics
    # MONAI metrics expect shape: [batch, num_classes, H, W, D] or [batch, H, W, D] for single class
    
    # Convert predictions and labels to one-hot format if they're not already
    if pred_tensor.dim() == 4:  # [batch, H, W, D] - single channel
        # Convert to one-hot format
        pred_one_hot = torch.zeros((pred_tensor.shape[0], num_classes, *pred_tensor.shape[1:]), 
                                  dtype=pred_tensor.dtype, device=pred_tensor.device)
        label_one_hot = torch.zeros((label_tensor.shape[0], num_classes, *label_tensor.shape[1:]), 
                                   dtype=label_tensor.dtype, device=label_tensor.device)
        
        for class_idx in range(num_classes):
            pred_one_hot[:, class_idx] = (pred_tensor.squeeze(1) == class_idx).float()
            label_one_hot[:, class_idx] = (label_tensor.squeeze(1) == class_idx).float()
        
        pred_tensor_metric = pred_one_hot
        label_tensor_metric = label_one_hot
    else:
        pred_tensor_metric = pred_tensor
        label_tensor_metric = label_tensor
    
    # Calculate Dice scores
    try:
        dice_scores = dice_metric(pred_tensor_metric, label_tensor_metric)
        if isinstance(dice_scores, torch.Tensor):
            if dice_scores.dim() == 0:  # scalar
                dice_scores_np = [dice_scores.item()]
            elif dice_scores.dim() == 1:  # [num_classes]
                dice_scores_np = dice_scores.cpu().numpy()
            else:  # [batch, num_classes] - take mean across batch
                dice_scores_np = dice_scores.mean(dim=0).cpu().numpy()
        else:
            dice_scores_np = np.array([dice_scores])
    except Exception as e:
        print(f"Error calculating dice scores: {e}")
        dice_scores_np = np.zeros(num_classes)

    # Calculate Hausdorff Distance
    try:
        hd_scores = hd_metric(pred_tensor_metric, label_tensor_metric)
        if isinstance(hd_scores, torch.Tensor):
            if hd_scores.dim() == 0:  # scalar
                hd_scores_np = [hd_scores.item()]
            elif hd_scores.dim() == 1:  # [num_classes]
                hd_scores_np = hd_scores.cpu().numpy()
            else:  # [batch, num_classes] - take mean across batch
                hd_scores_np = hd_scores.mean(dim=0).cpu().numpy()
        else:
            hd_scores_np = np.array([hd_scores])
    except Exception as e:
        print(f"Error calculating hausdorff distance: {e}")
        hd_scores_np = np.full(num_classes, float('inf'))

    # Ensure we have the right number of classes
    if len(dice_scores_np) != num_classes:
        dice_scores_np = np.pad(dice_scores_np, (0, max(0, num_classes - len(dice_scores_np))), 'constant')[:num_classes]
    if len(hd_scores_np) != num_classes:
        hd_scores_np = np.pad(hd_scores_np, (0, max(0, num_classes - len(hd_scores_np))), 'constant', constant_values=float('inf'))[:num_classes]

    # Convert to numpy for manual calculations
    pred_np = pred_tensor.squeeze().cpu().numpy()
    label_np = label_tensor.squeeze().cpu().numpy()

    # Metrics dicts
    per_class = {}
    for class_idx in range(num_classes):
        pred_bin = (pred_np == class_idx).astype(np.uint8)
        label_bin = (label_np == class_idx).astype(np.uint8)

        tp = np.sum((pred_bin == 1) & (label_bin == 1))
        tn = np.sum((pred_bin == 0) & (label_bin == 0))
        fp = np.sum((pred_bin == 1) & (label_bin == 0))
        fn = np.sum((pred_bin == 0) & (label_bin == 1))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        # Safe conversion to scalar
        dice_val = float(dice_scores_np[class_idx]) if class_idx < len(dice_scores_np) else 0.0
        hd_val = float(hd_scores_np[class_idx]) if class_idx < len(hd_scores_np) else float('inf')

        per_class[f"class_{class_idx}"] = {
            "dice": round(dice_val, 4),
            "hausdorff": round(hd_val, 4) if hd_val != float('inf') else "inf",
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
            "iou": round(iou, 4),
            "true_positive": int(tp),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
        }

    # Overall metrics (mean) - exclude infinite values
    valid_dice = dice_scores_np[np.isfinite(dice_scores_np)]
    valid_hd = hd_scores_np[np.isfinite(hd_scores_np)]
    
    overall = {
        "mean_dice": round(np.mean(valid_dice), 4) if len(valid_dice) > 0 else 0.0,
        "mean_hausdorff": round(np.mean(valid_hd), 4) if len(valid_hd) > 0 else "inf"
    }

    return {
        "overall": overall,
        "per_class": per_class
    }


import matplotlib.pyplot as plt
import torch
import numpy as np

def show_slice(image_tensor, label_tensor, pred_tensor, slice_index=120, num_classes=4):
    """
    Display one slice (axial) from 3D volumes: image, label, prediction.
    Assumes shape: [1, C, H, W, D]
    """
    # Squeeze batch and channel dims
    image = image_tensor[0, 0].cpu().numpy()      # [H, W, D]
    label = label_tensor[0, 0].cpu().numpy()      # [H, W, D]
    
    # For prediction, get predicted class per voxel
    pred_class = torch.argmax(pred_tensor, dim=1)  # [1, H, W, D]
    pred = pred_class[0].cpu().numpy()             # [H, W, D]

    # Extract the slice
    image_slice = image[:, :, slice_index]
    label_slice = label[:, :, slice_index]
    pred_slice = pred[:, :, slice_index]

    # Plot
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_slice, cmap='gray')
    plt.title(f"Image Slice {slice_index}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(label_slice, cmap='jet', vmin=0, vmax=num_classes - 1)
    plt.title(f"Ground Truth Slice {slice_index}")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_slice, cmap='jet', vmin=0, vmax=num_classes - 1)
    plt.title(f"Prediction Slice {slice_index}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

import torch

def print_slice_values(image_tensor, label_tensor, pred_tensor, slice_index=120):
    """
    Print raw values of image, label, and predicted segmentation at a given slice index.
    Assumes tensors of shape:
    - image_tensor: [1, 1, H, W, D]
    - label_tensor: [1, 1, H, W, D]
    - pred_tensor: [1, C, H, W, D]  (C = number of classes)
    """

    # Remove batch and channel dims
    image_np = image_tensor[0, 0].cpu().numpy()  # [H, W, D]
    label_np = label_tensor[0, 0].cpu().numpy()  # [H, W, D]

    # Get predicted class per voxel
    pred_class = torch.argmax(pred_tensor, dim=1)  # [1, H, W, D]
    pred_np = pred_class[0].cpu().numpy()          # [H, W, D]

    # Get the slice (axial direction assumed: D is last)
    image_slice = image_np[:, :, slice_index]
    label_slice = label_np[:, :, slice_index]
    pred_slice = pred_np[:, :, slice_index]

    # Print values
    print(f"\n=== Image Slice [{slice_index}] Values ===")
    print(image_slice)

    print(f"\n=== Label Slice [{slice_index}] Values ===")
    print(label_slice.astype(int))

    print(f"\n=== Prediction Slice [{slice_index}] Values ===")
    print(pred_slice.astype(int))

    # Optional: Print unique values in prediction
    print(f"\nUnique values in prediction slice: {np.unique(pred_slice)}")
import torch
import numpy as np

def save_slice_values_to_file(image_tensor, label_tensor, pred_tensor, slice_index=120, output_path="slice_120_values.txt"):
    """
    Save image, label, and prediction values of a specific slice to a text file.
    
    Args:
        image_tensor: torch.Tensor of shape [1, 1, H, W, D]
        label_tensor: torch.Tensor of shape [1, 1, H, W, D]
        pred_tensor: torch.Tensor of shape [1, C, H, W, D]
        slice_index: int, the index of the slice to extract
        output_path: str, path to save the output file
    """

    # Convert tensors to NumPy
    image_np = image_tensor[0, 0].cpu().numpy()        # [H, W, D]
    label_np = label_tensor[0, 0].cpu().numpy()        # [H, W, D]
    pred_np = torch.argmax(pred_tensor, dim=1)[0].cpu().numpy()  # [H, W, D]

    # Get slice values
    image_slice = image_np[:, :, slice_index]
    label_slice = label_np[:, :, slice_index].astype(int)
    pred_slice = pred_np[:, :, slice_index].astype(int)

    # Save to file
    with open(output_path, 'w') as f:
        f.write(f"=== Image Slice [{slice_index}] ===\n")
        np.savetxt(f, image_slice, fmt="%.4f", delimiter=" ")
        
        f.write(f"\n=== Label Slice [{slice_index}] ===\n")
        np.savetxt(f, label_slice, fmt="%d", delimiter=" ")
        
        f.write(f"\n=== Prediction Slice [{slice_index}] ===\n")
        np.savetxt(f, pred_slice, fmt="%d", delimiter=" ")

        # Print unique values
        f.write(f"\nUnique values in prediction slice: {np.unique(pred_slice)}\n")
    
    print(f"Slice data saved to: {output_path}")



def run_inference(model_path, image_path, label_path, pred_path, model_name="unetr", device=None):
    """
    Run inference on a single image and calculate metrics
    
    Args:
        model_path: path to trained model
        image_path: path to input image
        label_path: path to ground truth label
        pred_path: path to save prediction
        model_name: name of the model architecture
        device: device to use for inference
    """
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = create_qct_segmentation_model(model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from: {model_path}")
    
    # Prepare data
    data_dict = {"image": image_path, "label": label_path}
    transforms = create_inference_transforms()
    
    # Apply transforms
    data = transforms(data_dict)
    
    # Add batch dimension
    image_tensor = data["image"].unsqueeze(0).to(device)
    label_tensor = data["label"].unsqueeze(0).to(device)
    
    print(f"Input image shape: {image_tensor.shape}")
    print(f"Input label shape: {label_tensor.shape}")
    
    # Run inference
    with torch.no_grad():
        # Use sliding window inference for large volumes
        pred_tensor = sliding_window_inference(
            inputs=image_tensor, 
            roi_size=(96, 96, 96), 
            sw_batch_size=1, 
            predictor=model,
            overlap=0.5
        )
        
        # Apply post-processing (convert to discrete values)
        post_pred = AsDiscrete(threshold=0.5)
        post_label = AsDiscrete(threshold=0.5)
        
        pred_tensor = post_pred(pred_tensor)
        label_tensor = post_label(label_tensor)
    
    print("Inference completed")
    print(pred_tensor.shape)

    print(image_tensor.shape)
    print(label_tensor.shape)
    print(pred_tensor.shape)
    print_slice_values(image_tensor, label_tensor, pred_tensor, slice_index=120)

    save_slice_values_to_file(image_tensor, label_tensor, pred_tensor, slice_index=120, output_path="slice_120_output.txt")


    
    # Convert to numpy for saving
    pred_np = pred_tensor.squeeze().cpu().numpy()
    
    # Save prediction as NIfTI
    if pred_path:
        save_nifti_prediction(pred_np, image_path, pred_path)
    
    # Calculate metrics
    metrics = calculate_metrics(pred_tensor, label_tensor)
    
    return pred_np, metrics

if __name__ == "__main__":
    # Set paths
    model_path = "/home/user/auto-annotation/auto-annotation/models/unetr/combined_model.pth"
    image_path = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/image/CT2/RG018_right_CT2.nii.gz"
    label_path = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/label/CT2/RG018_right_CT2.nii.gz"
    
    # Create prediction path based on image filename
    base_name = os.path.splitext(os.path.splitext(os.path.basename(image_path))[0])[0]  # Remove .nii.gz
    pred_path = f"/home/user/auto-annotation/auto-annotation/predictions/{base_name}_prediction.nii.gz"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run inference
    try:
        prediction, metrics = run_inference(
            model_path=model_path,
            image_path=image_path,
            label_path=label_path,
            pred_path=pred_path,
            model_name="unetr",
            device=device
        )
        
        # Print results
        print("\n" + "="*60)
        print("MULTI-CLASS SEGMENTATION METRICS")
        print("="*60)
        
        # Print overall metrics
        print("\nOVERALL METRICS:")
        print("-" * 30)
        for metric_name, value in metrics['overall'].items():
            print(f"{metric_name.replace('_', ' ').title()}: {value}")
        
        # Print per-class metrics
        print("\nPER-CLASS METRICS:")
        print("-" * 30)
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"\n{class_name}:")
            for metric_name, value in class_metrics.items():
                print(f"  {metric_name.replace('_', ' ').title()}: {value}")
        
        print(f"\nPrediction shape: {prediction.shape}")
        print(f"Unique values in prediction: {np.unique(prediction)}")
        print(f"Prediction saved to: {pred_path}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()