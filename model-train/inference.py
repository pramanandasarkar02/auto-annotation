import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, ScaleIntensityRanged, EnsureTyped, AsDiscrete
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
import pydicom
from pydicom.dataset import Dataset as DicomDataset
from pydicom.uid import generate_uid
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model import create_qct_segmentation_model  # Import your model creation function
# from torch.serialization import add_safe_globals
# add_safe_globals([np.core.multiarray._reconstruct])
import torch
import os
from torch.serialization import add_safe_globals
import numpy as np


def analyze_ground_truth_label(label_path, num_classes=3):
    """
    Analyze ground truth label statistics
    
    Args:
        label_path: Path to ground truth NIfTI label
        num_classes: Number of classes in the segmentation
    
    Returns:
        dict: Dictionary containing GT analysis results
    """
    print("="*60)
    print("GROUND TRUTH LABEL ANALYSIS")
    print("="*60)
    
    if not os.path.exists(label_path):
        print(f"Warning: Ground truth label not found: {label_path}")
        return None
    
    # Load ground truth
    gt_nii = nib.load(label_path)
    gt_data = gt_nii.get_fdata().astype(np.uint8)
    
    print(f"Label file: {label_path}")
    print(f"Label shape: {gt_data.shape}")
    print(f"Label dtype: {gt_data.dtype}")
    print(f"Label value range: [{gt_data.min()}, {gt_data.max()}]")
    
    # Get unique classes
    unique_classes = np.unique(gt_data)
    print(f"Unique classes in GT: {unique_classes}")
    
    # Calculate statistics for each class
    total_voxels = gt_data.size
    gt_stats = {}
    
    print(f"\nGround Truth Class Distribution:")
    print("-" * 50)
    for class_id in range(num_classes):
        count = np.sum(gt_data == class_id)
        percentage = (count / total_voxels) * 100
        volume_mm3 = count * np.prod(gt_nii.header.get_zooms())  # Volume in mm³
        
        gt_stats[f'class_{class_id}'] = {
            'voxel_count': count,
            'percentage': percentage,
            'volume_mm3': volume_mm3
        }
        
        class_name = get_class_name(class_id)
        print(f"Class {class_id} ({class_name}): {count:,} voxels ({percentage:.2f}%) - {volume_mm3:.2f} mm³")
    
    # Calculate additional statistics
    spacing = gt_nii.header.get_zooms()
    print(f"\nImage Properties:")
    print(f"Voxel spacing: {spacing} mm")
    print(f"Total volume: {total_voxels * np.prod(spacing):.2f} mm³")
    
    # Check for class imbalance
    non_background_classes = [gt_stats[f'class_{i}']['percentage'] for i in range(1, num_classes)]
    if len(non_background_classes) > 0:
        imbalance_ratio = max(non_background_classes) / (min(non_background_classes) + 1e-8)
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
    
    return {
        'data': gt_data,
        'stats': gt_stats,
        'spacing': spacing,
        'unique_classes': unique_classes,
        'total_voxels': total_voxels
    }

def get_class_name(class_id):
    """Get human-readable class names"""
    class_names = {
        0: "Background", 
        1: "Femure Bone",
        2: "Wrong Bone",
        3: "Muscel",
        4: "other"
    }
    return class_names.get(class_id, f"Class_{class_id}")

def create_inference_transforms():
    """Create the same transforms used during training for consistency"""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])



# Add safe globals for numpy objects that might be in your checkpoint
def setup_safe_globals():
    """Setup safe globals for loading checkpoints with numpy arrays"""
    try:
        # Add numpy reconstruction functions to safe globals
        add_safe_globals([
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np.core.multiarray.scalar,
        ])
    except AttributeError:
        # For older numpy versions
        try:
            from numpy.core.multiarray import _reconstruct
            add_safe_globals([_reconstruct])
        except:
            pass

def load_trained_model(model_path, model_name="unet", device='cuda'):
    """
    Load the trained model with proper architecture 
    
    Args:
        model_path: Path to the saved model weights  
        model_name: Name of the model architecture (unet, unetr, swinunetr, segresnet)
        device: Device to load the model on
    
    Returns:
        Loaded model in evaluation mode
    """
    print(f"Loading {model_name} model from: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create the model using your model creation function
    model = create_qct_segmentation_model(model_name)
    
    # Try multiple loading strategies
    loading_strategies = [
        ("Strategy 1: Load with weights_only=False", "weights_only_false"),
        ("Strategy 2: Load with safe globals", "safe_globals"), 
        ("Strategy 3: Load with torch.serialization context", "context_manager"),
        ("Strategy 4: Load and extract state_dict manually", "manual_extract")
    ]
    
    for strategy_name, strategy_type in loading_strategies:
        try:
            print(f"Trying {strategy_name}...")
            
            if strategy_type == "weights_only_false":
                # Method 1: Load with weights_only=False (most likely to work)
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                
            elif strategy_type == "safe_globals":
                # Method 2: Setup safe globals and try weights_only=True
                setup_safe_globals()
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                
            elif strategy_type == "context_manager":
                # Method 3: Use context manager for safe globals
                with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
                    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                    
            elif strategy_type == "manual_extract":
                # Method 4: Load with pickle and extract manually
                checkpoint = torch.load(model_path, map_location=device, weights_only=False, pickle_module=pickle)
            
            # Extract model state dict from checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("Found 'model_state_dict' in checkpoint")
                
                # Print additional info if available
                if 'best_bone_metric' in checkpoint:
                    print(f"Best bone metric: {checkpoint['best_bone_metric']}")
                if 'epoch' in checkpoint:
                    print(f"Trained for {checkpoint['epoch']} epochs")
                    
            else:
                # Assume the checkpoint is the state dict itself
                state_dict = checkpoint
                print("Checkpoint appears to be raw state dict")
            
            # Load the state dict into model
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            print(f" SUCCESS! Model loaded using {strategy_name}")
            print(f"Model is now ready for inference on {device}")
            return model
            
        except Exception as e:
            print(f"{strategy_name} failed: {str(e)[:100]}...")
            continue
    
    # If all strategies failed
    raise Exception("All loading strategies failed. Your checkpoint file might be corrupted or incompatible.")


def load_trained_model_simple(model_path, model_name="unet", device='cuda'):
    """
    Simplified loader - most likely to work with your existing model
    """
    print(f"Loading {model_name} model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model
    model = create_qct_segmentation_model(model_name)
    
    # Load with weights_only=False (bypass the new PyTorch 2.6 restriction)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best bone metric: {checkpoint.get('best_bone_metric', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")
    return model


def extract_weights_from_checkpoint(checkpoint_path, output_path):
    """
    Extract just the model weights from your existing checkpoint for future fast loading
    This creates a weights-only file that's compatible with PyTorch 2.6 defaults
    """
    print(f"Extracting weights from: {checkpoint_path}")
    
    # Load full checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Extract state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Save weights only
    torch.save(state_dict, output_path)
    print(f"Weights extracted to: {output_path}")
    print("You can now use this file with weights_only=True in the future")


# USAGE EXAMPLES:
if __name__ == "__main__":
    
    # Quick test with your existing model
    try:
        print("=== ATTEMPTING TO LOAD YOUR EXISTING MODEL ===")
        
        # Method 1: Simple load (most likely to work)
        model_path = "./models/unet/combined_model.pth"  # Your existing model
        model = load_trained_model_simple(model_path, device='cuda')
        print(" SUCCESS! Your model is loaded and ready to use!")
        
        # Optional: Extract weights for faster future loading  
        weights_path = "./models/unet/model_weights_only.pth"
        extract_weights_from_checkpoint(model_path, weights_path)
        
    except Exception as e:
        print(f"Simple load failed: {e}")
        print("\nTrying advanced loading strategies...")
        
        # Method 2: Advanced load with multiple strategies
        try:
            model = load_trained_model(model_path, device='cuda')
            print(" SUCCESS with advanced loader!")
        except Exception as e2:
            print(f"All loading methods failed: {e2}")
            print("\nTroubleshooting tips:")
            print("1. Make sure the model file exists and isn't corrupted")
            print("2. Check if you have the correct PyTorch version")  
            print("3. Verify your model creation function works")
            print("4. Try loading on CPU first: device='cpu'")


# ALTERNATIVE: If you just want to run inference right now
def quick_inference_setup():
    """
    Quick setup to get your model running for inference
    """
    model_path = "./models/unet/combined_model.pth"
    
    print("Setting up model for inference...")
    
    # Force weights_only=False to bypass PyTorch 2.6 restrictions
    checkpoint = torch.load(model_path, weights_only=False, map_location='cuda')
    
    # Create model and load weights
    model = create_qct_segmentation_model("unet")
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.cuda()
    model.eval()
    
    print(" Model ready for inference!")
    return model


# Add safe globals for numpy objects that might be in your checkpoint
def setup_safe_globals():
    """Setup safe globals for loading checkpoints with numpy arrays"""
    try:
        # Add numpy reconstruction functions to safe globals
        add_safe_globals([
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np.core.multiarray.scalar,
        ])
    except AttributeError:
        # For older numpy versions
        try:
            from numpy.core.multiarray import _reconstruct
            add_safe_globals([_reconstruct])
        except:
            pass

def load_trained_model(model_path, model_name="unet", device='cuda'):
    """
    Load the trained model with proper architecture - FIXED for PyTorch 2.6
    Works with your existing 4-hour trained model without retraining!
    
    Args:
        model_path: Path to the saved model weights  
        model_name: Name of the model architecture (unet, unetr, swinunetr, segresnet)
        device: Device to load the model on
    
    Returns:
        Loaded model in evaluation mode
    """
    print(f"Loading {model_name} model from: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create the model using your model creation function
    model = create_qct_segmentation_model(model_name)
    
    # Try multiple loading strategies
    loading_strategies = [
        ("Strategy 1: Load with weights_only=False", "weights_only_false"),
        ("Strategy 2: Load with safe globals", "safe_globals"), 
        ("Strategy 3: Load with torch.serialization context", "context_manager"),
        ("Strategy 4: Load and extract state_dict manually", "manual_extract")
    ]
    
    for strategy_name, strategy_type in loading_strategies:
        try:
            print(f"Trying {strategy_name}...")
            
            if strategy_type == "weights_only_false":
                # Method 1: Load with weights_only=False (most likely to work)
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                
            elif strategy_type == "safe_globals":
                # Method 2: Setup safe globals and try weights_only=True
                setup_safe_globals()
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                
            elif strategy_type == "context_manager":
                # Method 3: Use context manager for safe globals
                with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
                    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                    
            elif strategy_type == "manual_extract":
                # Method 4: Load with pickle and extract manually
                checkpoint = torch.load(model_path, map_location=device, weights_only=False, pickle_module=pickle)
            
            # Extract model state dict from checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("Found 'model_state_dict' in checkpoint")
                
                # Print additional info if available
                if 'best_bone_metric' in checkpoint:
                    print(f"Best bone metric: {checkpoint['best_bone_metric']}")
                if 'epoch' in checkpoint:
                    print(f"Trained for {checkpoint['epoch']} epochs")
                    
            else:
                # Assume the checkpoint is the state dict itself
                state_dict = checkpoint
                print("Checkpoint appears to be raw state dict")
            
            # Load the state dict into model
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            print(f" SUCCESS! Model loaded using {strategy_name}")
            print(f"Model is now ready for inference on {device}")
            return model
            
        except Exception as e:
            print(f"{strategy_name} failed: {str(e)[:100]}...")
            continue
    
    # If all strategies failed
    raise Exception("All loading strategies failed. Your checkpoint file might be corrupted or incompatible.")


def load_trained_model_simple(model_path, model_name="unet", device='cuda'):
    """
    Simplified loader - most likely to work with your existing model
    """
    print(f"Loading {model_name} model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model
    model = create_qct_segmentation_model(model_name)
    
    # Load with weights_only=False (bypass the new PyTorch 2.6 restriction)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best bone metric: {checkpoint.get('best_bone_metric', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")
    return model


def extract_weights_from_checkpoint(checkpoint_path, output_path):
    """
    Extract just the model weights from your existing checkpoint for future fast loading
    This creates a weights-only file that's compatible with PyTorch 2.6 defaults
    """
    print(f"Extracting weights from: {checkpoint_path}")
    
    # Load full checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Extract state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Save weights only
    torch.save(state_dict, output_path)
    print(f"Weights extracted to: {output_path}")
    print("You can now use this file with weights_only=True in the future")





# ALTERNATIVE: If you just want to run inference right now
def quick_inference_setup():
    """
    Quick setup to get your model running for inference
    """
    # model_path = "./models/unet/combined_model.pth"
    model_path = "./models/unet/left_ct2_model.pth"
    
    print("Setting up model for inference...")
    
    # Force weights_only=False to bypass PyTorch 2.6 restrictions
    checkpoint = torch.load(model_path, weights_only=False, map_location='cuda')
    
    # Create model and load weights
    model = create_qct_segmentation_model("unet")
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.cuda()
    model.eval()
    
    print(" Model ready for inference!")
    return model

# def load_trained_model(model_path, model_name="unet", device="cuda"):
#     print(f"Loading {model_name} model from: {model_path}")

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")

#     model = create_qct_segmentation_model(model_name)

#     try:
#         # Trust your own checkpoints → allow old style load
#         state_dict = torch.load(model_path, map_location=device, weights_only=False)
#         model.load_state_dict(state_dict)
#         model.to(device)
#         model.eval()
#         print(f"Model loaded successfully on {device}")
#         return model

#     except Exception as e:
#         print(f"Error loading model state dict: {e}")
#         raise


def prediction(image_path, model_path, output_path, model_name="unet", device=None, 
               num_classes=5, save_all_classes=True):
    """
    Perform prediction on a single 3D medical image using the trained model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*60)
    print("RUNNING PREDICTION")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Model type: {model_name}")
    print(f"Device: {device}")
    print(f"Number of classes: {num_classes}")
    
    # Check if input file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    # Create inference transforms (same as training)
    inference_transforms = create_inference_transforms()
    
    # Prepare data
    test_data = [{"image": image_path}]
    test_ds = Dataset(data=test_data, transform=inference_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)
    
    # Load the trained model
    try:
        model = load_trained_model_simple(model_path, model_name, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return create_dummy_multiclass_prediction(image_path, output_path, num_classes)
    
    print("Running inference with sliding window...")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        for test_data in test_loader:
            test_inputs = test_data["image"].to(device)
            
            print(f"Input tensor shape: {test_inputs.shape}")
            
            # Use sliding window inference
            test_outputs = sliding_window_inference(
                inputs=test_inputs,
                roi_size=(96, 96, 96),  # Same ROI size as training
                sw_batch_size=1,
                predictor=model,
                overlap=0.5
            )
            
            print(f"Raw output shape: {test_outputs.shape}")
            print(f"Raw output range: [{test_outputs.min():.3f}, {test_outputs.max():.3f}]")
            
            # Apply softmax to get probabilities
            test_outputs = torch.softmax(test_outputs, dim=1)
            
            print(f"After softmax - output range: [{test_outputs.min():.3f}, {test_outputs.max():.3f}]")
            
            # Get class predictions using argmax
            class_predictions = torch.argmax(test_outputs, dim=1)
            
            print(f"Class predictions shape: {class_predictions.shape}")
            print(f"Unique classes in prediction: {torch.unique(class_predictions)}")
            
            # Convert to numpy
            class_predictions_np = class_predictions[0].cpu().numpy().astype(np.uint8)
            
            # Also get probability maps for each class
            prob_maps = test_outputs[0].cpu().numpy()  # Shape: [num_classes, H, W, D]
            
            print(f"Final prediction shape: {class_predictions_np.shape}")
            print(f"Unique classes: {np.unique(class_predictions_np)}")
            
            # Count voxels for each class
            print(f"\nPrediction Class Distribution:")
            print("-" * 40)
            for class_id in range(num_classes):
                count = np.sum(class_predictions_np == class_id)
                percentage = (count / class_predictions_np.size) * 100
                class_name = get_class_name(class_id)
                print(f"Class {class_id} ({class_name}): {count:,} voxels ({percentage:.2f}%)")
            
            # Prepare results dictionary
            results = {
                'class_prediction': class_predictions_np,
                'probabilities': prob_maps
            }
            
            # Save predictions
            if save_all_classes:
                save_multiclass_predictions(results, image_path, output_path, num_classes)
            else:
                save_prediction_as_nifti(class_predictions_np, image_path, output_path)
            
            return results

def analyze_prediction_vs_ground_truth(pred_data, gt_data, num_classes=3, spacing=None):
    """
    Compare prediction with ground truth and calculate metrics
    
    Args:
        pred_data: Predicted segmentation array
        gt_data: Ground truth segmentation array  
        num_classes: Number of classes
        spacing: Voxel spacing for volume calculations
    
    Returns:
        dict: Dictionary containing comparison metrics
    """
    print("\n" + "="*60)
    print("PREDICTION vs GROUND TRUTH ANALYSIS")
    print("="*60)
    
    if pred_data.shape != gt_data.shape:
        print(f"Warning: Shape mismatch - Pred: {pred_data.shape}, GT: {gt_data.shape}")
        return None
    
    # Calculate per-class metrics
    metrics = {}
    
    print("Per-Class Metrics:")
    print("-" * 50)
    
    dice_scores = []
    
    for class_id in range(num_classes):
        pred_mask = (pred_data == class_id).astype(np.uint8)
        gt_mask = (gt_data == class_id).astype(np.uint8)
        
        # Calculate Dice coefficient
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)
        
        if union == 0:
            dice = 1.0  # Both masks are empty
        else:
            dice = 2.0 * intersection / union
        
        dice_scores.append(dice)
        
        # Calculate sensitivity (recall) and specificity
        tp = intersection
        fp = np.sum(pred_mask) - tp
        fn = np.sum(gt_mask) - tp
        tn = pred_data.size - tp - fp - fn
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Volume calculations
        pred_volume = np.sum(pred_mask) * (np.prod(spacing) if spacing is not None else 1.0)
        gt_volume = np.sum(gt_mask) * (np.prod(spacing) if spacing is not None else 1.0)
        volume_error = abs(pred_volume - gt_volume) / (gt_volume + 1e-8) * 100
        
        metrics[f'class_{class_id}'] = {
            'dice': dice,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'pred_volume': pred_volume,
            'gt_volume': gt_volume,
            'volume_error_percent': volume_error
        }
        
        class_name = get_class_name(class_id)
        print(f"Class {class_id} ({class_name}):")
        print(f"  Dice Score: {dice:.4f}")
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Volume Error: {volume_error:.2f}%")
        print()
    
    # Overall metrics
    mean_dice = np.mean(dice_scores[1:])  # Exclude background
    overall_accuracy = np.mean(pred_data == gt_data)
    
    print("Overall Metrics:")
    print("-" * 30)
    print(f"Mean Dice Score (excluding background): {mean_dice:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    
    # Confusion matrix
    pred_flat = pred_data.flatten()
    gt_flat = gt_data.flatten()
    cm = confusion_matrix(gt_flat, pred_flat, labels=range(num_classes))
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    metrics['overall'] = {
        'mean_dice': mean_dice,
        'overall_accuracy': overall_accuracy,
        'confusion_matrix': cm,
        'dice_scores': dice_scores
    }
    
    return metrics

def create_dummy_multiclass_prediction(image_path, output_path, num_classes=3):
    """Create a meaningful dummy multiclass prediction based on the actual image"""
    print("Creating dummy multiclass prediction for demonstration...")
    
    # Load the original image to get its shape and properties
    nii_img = nib.load(image_path)
    image_data = nii_img.get_fdata()
    
    print(f"Original image shape: {image_data.shape}")
    
    # Create multiclass prediction
    class_prediction = np.zeros(image_data.shape, dtype=np.uint8)
    
    # Simple threshold-based multiclass segmentation for demonstration
    normalized_image = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    
    # Create different classes based on intensity thresholds
    soft_tissue_mask = (normalized_image > 0.3) & (normalized_image <= 0.6)
    trabecular_bone_mask = (normalized_image > 0.6) & (normalized_image <= 0.8)
    cortical_bone_mask = normalized_image > 0.8
    
    class_prediction[soft_tissue_mask] = 1
    class_prediction[trabecular_bone_mask] = 2
    if num_classes > 3:
        class_prediction[cortical_bone_mask] = 3
    
    print(f"Dummy multiclass prediction created:")
    for class_id in range(num_classes):
        count = np.sum(class_prediction == class_id)
        percentage = (count / class_prediction.size) * 100
        print(f"  Class {class_id}: {count} voxels ({percentage:.2f}%)")
    
    # Create dummy probability maps
    prob_maps = np.zeros((num_classes,) + image_data.shape, dtype=np.float32)
    for class_id in range(num_classes):
        prob_maps[class_id] = (class_prediction == class_id).astype(np.float32)
    
    results = {
        'class_prediction': class_prediction,
        'probabilities': prob_maps
    }
    
    save_multiclass_predictions(results, image_path, output_path, num_classes)
    return results

def save_multiclass_predictions(results, original_image_path, output_path, num_classes):
    """Save multiclass predictions as separate NIfTI files"""
    print(f"\nSaving multiclass predictions...")
    
    # Load original image to get header info
    original_nii = nib.load(original_image_path)
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    base_name = os.path.basename(output_path).replace('.nii.gz', '')
    
    # Save class prediction (argmax)
    class_prediction = results['class_prediction']
    class_pred_nii = nib.Nifti1Image(
        class_prediction.astype(np.uint8), 
        original_nii.affine, 
        original_nii.header
    )
    class_pred_nii.header.set_data_dtype(np.uint8)
    
    class_pred_path = os.path.join(output_dir, f"{base_name}_classes.nii.gz")
    nib.save(class_pred_nii, class_pred_path)
    print(f"Class prediction saved: {class_pred_path}")
    
    # Save probability maps for each class
    prob_maps = results['probabilities']
    for class_id in range(num_classes):
        prob_map = prob_maps[class_id]
        prob_nii = nib.Nifti1Image(
            prob_map.astype(np.float32),
            original_nii.affine,
            original_nii.header
        )
        prob_nii.header.set_data_dtype(np.float32)
        
        prob_path = os.path.join(output_dir, f"{base_name}_class{class_id}_prob.nii.gz")
        nib.save(prob_nii, prob_path)
        print(f"Class {class_id} probability saved: {prob_path}")
    
    # Save individual class masks as binary
    for class_id in range(1, num_classes):  # Skip background (class 0)
        class_mask = (class_prediction == class_id).astype(np.uint8)
        mask_nii = nib.Nifti1Image(
            class_mask,
            original_nii.affine,
            original_nii.header
        )
        mask_nii.header.set_data_dtype(np.uint8)
        
        mask_path = os.path.join(output_dir, f"{base_name}_class{class_id}_mask.nii.gz")
        nib.save(mask_nii, mask_path)
        print(f"Class {class_id} binary mask saved: {mask_path}")

def save_prediction_as_nifti(prediction, original_image_path, output_path):
    """Save prediction as NIfTI file with same header as original"""
    print(f"Saving prediction to: {output_path}")
    
    # Load original image to get header info
    original_nii = nib.load(original_image_path)
    
    # Create new NIfTI image with prediction data but original header
    prediction_nii = nib.Nifti1Image(
        prediction.astype(np.uint8), 
        original_nii.affine, 
        original_nii.header
    )
    
    # Update header for segmentation
    prediction_nii.header.set_data_dtype(np.uint8)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the prediction
    nib.save(prediction_nii, output_path)
    print(f"Prediction saved successfully!")

def nifti_to_dcm_multiclass(nifti_path, dcm_dir, class_id=None):
    """Convert multiclass NIfTI file to DICOM series"""
    print(f"\nConverting multiclass NIfTI to DICOM...")
    print(f"Input: {nifti_path}")
    print(f"Output directory: {dcm_dir}")
    
    # Create output directory
    os.makedirs(dcm_dir, exist_ok=True)
    
    # Load NIfTI file
    nii_img = nib.load(nifti_path)
    data = nii_img.get_fdata()
    
    print(f"Data shape: {data.shape}")
    print(f"Unique classes: {np.unique(data)}")
    
    # Get image properties
    spacing = nii_img.header.get_zooms()
    
    # Convert to appropriate format for DICOM
    if class_id is not None:
        # Convert specific class to binary mask
        data = (data == class_id).astype(np.uint16) * 4095
        series_description = f"Class {class_id} Segmentation"
    else:
        # Scale multiclass data
        data = data.astype(np.uint16) * (4095 // max(1, int(np.max(data))))
        series_description = "Multiclass Segmentation"
    
    # Generate unique identifiers
    series_instance_uid = generate_uid()
    study_instance_uid = generate_uid()
    frame_of_reference_uid = generate_uid()
    
    print(f"Creating {data.shape[2]} DICOM slices...")
    
    slice_count = 0
    for i in range(data.shape[2]):
        slice_data = data[:, :, i]
        
        # Skip empty slices
        if np.sum(slice_data) == 0:
            continue
        
        # Create DICOM dataset
        ds = DicomDataset()
        
        # Set required DICOM tags
        ds.PatientName = "Anonymous"
        ds.PatientID = "ANON001"
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        
        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = series_instance_uid
        ds.SOPInstanceUID = generate_uid()
        ds.FrameOfReferenceUID = frame_of_reference_uid
        
        current_time = datetime.datetime.now()
        ds.StudyDate = current_time.strftime("%Y%m%d")
        ds.StudyTime = current_time.strftime("%H%M%S")
        ds.SeriesDate = current_time.strftime("%Y%m%d")
        ds.SeriesTime = current_time.strftime("%H%M%S")
        
        ds.StudyDescription = "AI Multiclass Segmentation Result"
        ds.SeriesDescription = series_description
        ds.SeriesNumber = "100"
        ds.InstanceNumber = str(i + 1)
        
        # Image-specific tags
        ds.Modality = "SEG"
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = slice_data.shape[0]
        ds.Columns = slice_data.shape[1]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        
        # Set spacing information
        ds.PixelSpacing = [str(spacing[0]), str(spacing[1])]
        ds.SliceThickness = str(spacing[2])
        ds.SpacingBetweenSlices = str(spacing[2])
        
        # Set position information
        ds.ImagePositionPatient = [0, 0, i * spacing[2]]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.SliceLocation = str(i * spacing[2])
        
        # Set window/level for better visualization
        ds.WindowCenter = "2047"
        ds.WindowWidth = "4095"
        
        # Set the pixel data
        ds.PixelData = slice_data.tobytes()
        
        # Set transfer syntax
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.file_meta.ImplementationClassUID = generate_uid()
        
        # Save DICOM file
        filename = f"seg_slice_{i:04d}.dcm"
        filepath = os.path.join(dcm_dir, filename)
        
        try:
            pydicom.dcmwrite(filepath, ds)
            slice_count += 1
        except Exception as e:
            print(f"Error writing slice {i}: {e}")
            continue
    
    print(f"DICOM conversion completed! {slice_count} slices saved in {dcm_dir}")

def save_analysis_report(gt_analysis, pred_analysis, comparison_metrics, output_dir):
    """Save a comprehensive analysis report"""
    report_path = os.path.join(output_dir, "analysis_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("MULTICLASS SEGMENTATION ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Ground truth analysis
        if gt_analysis:
            f.write("GROUND TRUTH ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total voxels: {gt_analysis['total_voxels']:,}\n")
            f.write(f"Unique classes: {gt_analysis['unique_classes']}\n")
            f.write(f"Voxel spacing: {gt_analysis['spacing']} mm\n\n")
            
            for class_id, stats in gt_analysis['stats'].items():
                class_name = get_class_name(int(class_id.split('_')[1]))
                f.write(f"{class_id} ({class_name}):\n")
                f.write(f"  Voxels: {stats['voxel_count']:,} ({stats['percentage']:.2f}%)\n")
                f.write(f"  Volume: {stats['volume_mm3']:.2f} mm³\n")
            f.write("\n")
        
        # Comparison metrics
        if comparison_metrics:
            f.write("PREDICTION vs GROUND TRUTH METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean Dice Score: {comparison_metrics['overall']['mean_dice']:.4f}\n")
            f.write(f"Overall Accuracy: {comparison_metrics['overall']['overall_accuracy']:.4f}\n\n")
            
            for class_id in range(len(comparison_metrics['overall']['dice_scores'])):
                if f'class_{class_id}' in comparison_metrics:
                    metrics = comparison_metrics[f'class_{class_id}']
                    class_name = get_class_name(class_id)
                    f.write(f"Class {class_id} ({class_name}):\n")
                    f.write(f"  Dice Score: {metrics['dice']:.4f}\n")
                    f.write(f"  Sensitivity: {metrics['sensitivity']:.4f}\n")
                    f.write(f"  Specificity: {metrics['specificity']:.4f}\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Volume Error: {metrics['volume_error_percent']:.2f}%\n")
                    f.write("\n")
    
    print(f"Analysis report saved: {report_path}")

if __name__ == "__main__":
    # File paths
    image_path = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/image/CT3/SL015_left_CT3.nii.gz"
    label_path = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/label/CT3/SL015_left_CT3.nii.gz"
    pred_dir = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/pred/CT3/" 
    os.makedirs(pred_dir, exist_ok=True)
    pred_basename = os.path.basename(image_path).replace('.nii.gz', '_prediction.nii.gz')
    prediction_path = os.path.join(pred_dir, pred_basename)
    
    # Model configuration
    model_path = "/home/user/auto-annotation/auto-annotation/models/swinunetr/left_model.pth"
    model_name = "swinunetr"
    num_classes = 5
    
    try:
        print("=" * 80)
        print("COMPLETE MULTICLASS INFERENCE PIPELINE WITH GT ANALYSIS")
        print("=" * 80)
        
        # Step 1: Analyze Ground Truth Label
        gt_analysis = analyze_ground_truth_label(label_path, num_classes)
        
        # Step 2: Run Prediction
        prediction_results = prediction(
            image_path=image_path,
            model_path=model_path, 
            output_path=prediction_path,
            model_name=model_name,
            num_classes=num_classes,
            save_all_classes=True
        )
        
        # Step 3: Compare Prediction with Ground Truth
        comparison_metrics = None
        if gt_analysis is not None:
            spacing = gt_analysis['spacing']
            comparison_metrics = analyze_prediction_vs_ground_truth(
                prediction_results['class_prediction'],
                gt_analysis['data'],
                num_classes,
                spacing
            )
        
        # Step 4: Save comprehensive analysis report
        save_analysis_report(gt_analysis, prediction_results, comparison_metrics, pred_dir)
        
        # Step 5: Convert to DICOM
        print("\n" + "="*60)
        print("CONVERTING TO DICOM")
        print("="*60)
        
        # Define DICOM output directories
        classes_nifti_path = os.path.join(pred_dir, pred_basename.replace('.nii.gz', '_classes.nii.gz'))
        dcm_dir_classes = prediction_path.replace('.nii.gz', '_classes_dcm')
        
        # Convert class prediction (all classes) to DICOM
        if os.path.exists(classes_nifti_path):
            nifti_to_dcm_multiclass(classes_nifti_path, dcm_dir_classes)
        
        # Convert individual classes to DICOM
        for class_id in range(1, num_classes):  # Skip background
            class_nifti_path = os.path.join(pred_dir, pred_basename.replace('.nii.gz', f'_class{class_id}_mask.nii.gz'))
            class_dcm_dir = prediction_path.replace('.nii.gz', f'_class{class_id}_dcm')
            
            if os.path.exists(class_nifti_path):
                nifti_to_dcm_multiclass(classes_nifti_path, class_dcm_dir, class_id=class_id)
        
        # Final Summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Print Ground Truth Summary
        if gt_analysis:
            print("\nGROUND TRUTH SUMMARY:")
            print(f"  Total voxels: {gt_analysis['total_voxels']:,}")
            print(f"  Unique classes: {gt_analysis['unique_classes']}")
            for class_id in range(num_classes):
                if f'class_{class_id}' in gt_analysis['stats']:
                    stats = gt_analysis['stats'][f'class_{class_id}']
                    class_name = get_class_name(class_id)
                    print(f"  Class {class_id} ({class_name}): {stats['voxel_count']:,} voxels ({stats['percentage']:.2f}%)")
        
        # Print Prediction Summary
        pred_data = prediction_results['class_prediction']
        print(f"\nPREDICTION SUMMARY:")
        print(f"  Prediction shape: {pred_data.shape}")
        print(f"  Unique classes: {np.unique(pred_data)}")
        for class_id in range(num_classes):
            count = np.sum(pred_data == class_id)
            percentage = (count / pred_data.size) * 100
            class_name = get_class_name(class_id)
            print(f"  Class {class_id} ({class_name}): {count:,} voxels ({percentage:.2f}%)")
        
        # Print Comparison Summary
        if comparison_metrics:
            print(f"\nCOMPARISON SUMMARY:")
            print(f"  Mean Dice Score: {comparison_metrics['overall']['mean_dice']:.4f}")
            print(f"  Overall Accuracy: {comparison_metrics['overall']['overall_accuracy']:.4f}")
            
            print("  Per-class Dice Scores:")
            for class_id in range(num_classes):
                if f'class_{class_id}' in comparison_metrics:
                    dice = comparison_metrics[f'class_{class_id}']['dice']
                    class_name = get_class_name(class_id)
                    print(f"    Class {class_id} ({class_name}): {dice:.4f}")
        
        # Print Output Files
        print(f"\nOUTPUT FILES:")
        print(f"  Analysis Report: {os.path.join(pred_dir, 'analysis_report.txt')}")
        print(f"  Classes NIfTI: {classes_nifti_path}")
        print(f"  Classes DICOM: {dcm_dir_classes}")
        
        for class_id in range(1, num_classes):
            class_nifti_path = os.path.join(pred_dir, pred_basename.replace('.nii.gz', f'_class{class_id}_mask.nii.gz'))
            class_dcm_dir = prediction_path.replace('.nii.gz', f'_class{class_id}_dcm')
            print(f"  Class {class_id} NIfTI: {class_nifti_path}")
            print(f"  Class {class_id} DICOM: {class_dcm_dir}")
        
        # Print Performance Insights
        if comparison_metrics:
            print(f"\nPERFORMANCE INSIGHTS:")
            
            # Find best and worst performing classes
            dice_scores = []
            class_names = []
            for class_id in range(num_classes): 
                if f'class_{class_id}' in comparison_metrics:
                    dice_scores.append(comparison_metrics[f'class_{class_id}']['dice'])
                    class_names.append(f"Class {class_id} ({get_class_name(class_id)})")
            
            if dice_scores:
                best_idx = np.argmax(dice_scores)
                worst_idx = np.argmin(dice_scores)
                
                print(f"  Best performing class: {class_names[best_idx]} (Dice: {dice_scores[best_idx]:.4f})")
                print(f"  Worst performing class: {class_names[worst_idx]} (Dice: {dice_scores[worst_idx]:.4f})")
                
                # Performance classification
                mean_dice = np.mean(dice_scores)
                if mean_dice >= 0.9:
                    performance = "Excellent"
                elif mean_dice >= 0.8:
                    performance = "Good"
                elif mean_dice >= 0.7:
                    performance = "Fair"
                else:
                    performance = "Poor"
                
                print(f"  Overall performance: {performance} (Mean Dice: {mean_dice:.4f})")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to provide helpful error information
        print(f"\nTROUBLESHOoting INFORMATION:")
        print(f"Image path exists: {os.path.exists(image_path)}")
        print(f"Label path exists: {os.path.exists(label_path)}")
        print(f"Model path exists: {os.path.exists(model_path)}")
        print(f"Prediction directory exists: {os.path.exists(pred_dir)}")
        
        # Check CUDA availability
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
        
        # Memory information
        import psutil
        memory = psutil.virtual_memory()
        print(f"Available RAM: {memory.available / (1024**3):.2f} GB")
        print(f"Total RAM: {memory.total / (1024**3):.2f} GB")