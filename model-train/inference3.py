import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, ScaleIntensityRanged, EnsureTyped
)
from monai.data import Dataset, DataLoader
from model import create_qct_segmentation_model

def create_exact_training_transforms():
    """
    Create EXACT same transforms used during training
    This is critical for consistent results
    """
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

def debug_preprocessing_pipeline(image_path, transforms):
    """
    Debug preprocessing to ensure it matches training
    """
    print("\n" + "="*60)
    print("DEBUGGING PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load raw image first
    raw_nii = nib.load(image_path)
    raw_data = raw_nii.get_fdata()
    
    print(f"Raw image stats:")
    print(f"  Shape: {raw_data.shape}")
    print(f"  Range: [{raw_data.min():.2f}, {raw_data.max():.2f}]")
    print(f"  Mean: {raw_data.mean():.2f}")
    print(f"  Std: {raw_data.std():.2f}")
    print(f"  Spacing: {raw_nii.header.get_zooms()}")
    
    # Apply transforms step by step
    test_data = [{"image": image_path}]
    test_ds = Dataset(data=test_data, transform=transforms)
    processed_data = test_ds[0]["image"]
    
    print(f"\nProcessed image stats:")
    print(f"  Shape: {processed_data.shape}")
    print(f"  Range: [{processed_data.min():.4f}, {processed_data.max():.4f}]")
    print(f"  Mean: {processed_data.mean():.4f}")
    print(f"  Std: {processed_data.std():.4f}")
    
    return processed_data

def improved_prediction_with_validation_mode(image_path, model_path, output_path, 
                                           model_name="unet", device=None, num_classes=5):
    """
    Improved prediction that mimics validation conditions exactly
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*60)
    print("IMPROVED PREDICTION WITH VALIDATION MODE")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    
    # Use EXACT same transforms as training
    transforms = create_exact_training_transforms()
    
    # Debug preprocessing
    processed_data = debug_preprocessing_pipeline(image_path, transforms)
    
    # Load model with improved error handling
    model = load_model_robust(model_path, model_name, device)
    
    # Prepare data exactly like validation
    test_data = [{"image": image_path}]
    test_ds = Dataset(data=test_data, transform=transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)
    
    model.eval()
    with torch.no_grad():
        for test_data_batch in test_loader:
            test_inputs = test_data_batch["image"].to(device)
            
            print(f"\nInput tensor stats:")
            print(f"  Shape: {test_inputs.shape}")
            print(f"  Range: [{test_inputs.min():.4f}, {test_inputs.max():.4f}]")
            print(f"  Mean: {test_inputs.mean():.4f}")
            print(f"  Device: {test_inputs.device}")
            
            # Use SAME sliding window parameters as training validation
            test_outputs = sliding_window_inference(
                inputs=test_inputs,
                roi_size=(64, 64, 64),  # SAME as training validation
                sw_batch_size=1,
                predictor=model,
                overlap=0.75  # SAME as training validation
            )
            
            print(f"\nModel output stats:")
            print(f"  Shape: {test_outputs.shape}")
            print(f"  Range: [{test_outputs.min():.4f}, {test_outputs.max():.4f}]")
            print(f"  Mean: {test_outputs.mean():.4f}")
            
            # Apply softmax
            test_outputs_softmax = torch.softmax(test_outputs, dim=1)
            
            print(f"\nAfter softmax:")
            print(f"  Range: [{test_outputs_softmax.min():.4f}, {test_outputs_softmax.max():.4f}]")
            
            # Check class-wise probabilities
            print("\nClass-wise max probabilities:")
            for class_id in range(num_classes):
                class_prob = test_outputs_softmax[0, class_id]
                print(f"  Class {class_id}: max={class_prob.max():.4f}, mean={class_prob.mean():.4f}")
            
            # Get predictions
            class_predictions = torch.argmax(test_outputs_softmax, dim=1)
            class_predictions_np = class_predictions[0].cpu().numpy().astype(np.uint8)
            
            # Analyze predictions
            print(f"\nPrediction analysis:")
            print(f"  Shape: {class_predictions_np.shape}")
            print(f"  Unique classes: {np.unique(class_predictions_np)}")
            
            total_voxels = class_predictions_np.size
            for class_id in range(num_classes):
                count = np.sum(class_predictions_np == class_id)
                percentage = (count / total_voxels) * 100
                print(f"  Class {class_id}: {count:,} voxels ({percentage:.2f}%)")
            
            # Save results
            save_prediction_with_probabilities(
                class_predictions_np, 
                test_outputs_softmax[0].cpu().numpy(),
                image_path, 
                output_path,
                num_classes
            )
            
            return {
                'class_prediction': class_predictions_np,
                'probabilities': test_outputs_softmax[0].cpu().numpy(),
                'raw_logits': test_outputs[0].cpu().numpy()
            }

def load_model_robust(model_path, model_name, device):
    """
    Robust model loading with multiple fallback strategies
    """
    print(f"Loading model: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Create model architecture
    model = create_qct_segmentation_model(model_name)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Training bone metric: {checkpoint.get('best_bone_metric', 'unknown')}")
        else:
            state_dict = checkpoint
        
        # Load weights
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def save_prediction_with_probabilities(prediction, probabilities, original_path, 
                                     output_path, num_classes):
    """
    Save prediction along with probability maps for analysis
    """
    original_nii = nib.load(original_path)
    output_dir = os.path.dirname(output_path)
    base_name = os.path.basename(output_path).replace('.nii.gz', '')
    
    # Save class prediction
    pred_nii = nib.Nifti1Image(
        prediction.astype(np.uint8),
        original_nii.affine,
        original_nii.header
    )
    pred_path = os.path.join(output_dir, f"{base_name}_classes.nii.gz")
    nib.save(pred_nii, pred_path)
    print(f"Class prediction saved: {pred_path}")
    
    # Save probability maps
    for class_id in range(num_classes):
        prob_map = probabilities[class_id]
        prob_nii = nib.Nifti1Image(
            prob_map.astype(np.float32),
            original_nii.affine,
            original_nii.header
        )
        prob_path = os.path.join(output_dir, f"{base_name}_class{class_id}_prob.nii.gz")
        nib.save(prob_nii, prob_path)
        print(f"Class {class_id} probability saved: {prob_path}")

def compare_with_training_validation(prediction_results, expected_distribution):
    """
    Compare inference results with training validation distribution
    """
    print("\n" + "="*60)
    print("COMPARING WITH TRAINING VALIDATION")
    print("="*60)
    
    pred_data = prediction_results['class_prediction']
    total_voxels = pred_data.size
    
    print("Class distribution comparison:")
    print(f"{'Class':<15} {'Training Val':<15} {'Inference':<15} {'Diff':<10}")
    print("-" * 65)
    
    for class_id in range(5):
        # Current inference distribution
        count = np.sum(pred_data == class_id)
        inference_pct = (count / total_voxels) * 100
        
        # Expected from training validation
        training_pct = expected_distribution.get(class_id, 0.0)
        
        # Difference
        diff = inference_pct - training_pct
        
        class_name = get_class_name(class_id)
        print(f"{class_name:<15} {training_pct:<15.2f} {inference_pct:<15.2f} {diff:<10.2f}")

def get_class_name(class_id):
    """Get human-readable class names"""
    class_names = {
        0: "Background", 
        1: "Bone",
        2: "Wrong Bone",
        3: "Muscle",
        4: "Extra Part"
    }
    return class_names.get(class_id, f"Class_{class_id}")

def diagnose_model_behavior(image_path, model_path, model_name="unet"):
    """
    Comprehensive diagnosis of model behavior
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL BEHAVIOR DIAGNOSIS")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run improved prediction
    results = improved_prediction_with_validation_mode(
        image_path, model_path, 
        image_path.replace('.nii.gz', '_diagnosed.nii.gz'),
        model_name, device
    )
    
    # Compare with expected training validation distribution
    expected_distribution = {
        0: 78.40,  # Background
        1: 0.56,   # Bone
        2: 0.44,   # Wrong Bone  
        3: 17.66,  # Muscle
        4: 2.94    # Extra Part
    }
    
    compare_with_training_validation(results, expected_distribution)
    
    # Additional analysis
    probabilities = results['probabilities']
    
    print(f"\nProbability analysis:")
    for class_id in range(5):
        prob_map = probabilities[class_id]
        print(f"Class {class_id} ({get_class_name(class_id)}):")
        print(f"  Max prob: {prob_map.max():.4f}")
        print(f"  Mean prob: {prob_map.mean():.4f}")
        print(f"  Std prob: {prob_map.std():.4f}")
        print(f"  Voxels >0.5: {np.sum(prob_map > 0.5):,}")
        print(f"  Voxels >0.9: {np.sum(prob_map > 0.9):,}")
    
    return results

# Usage example
if __name__ == "__main__":
    # Your paths
    image_path = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data/image/CT2/RT023_left_CT2.nii.gz"
    model_path = "/home/user/auto-annotation/auto-annotation/models/unet/combined_model.pth"
    
    # Run comprehensive diagnosis
    try:
        results = diagnose_model_behavior(image_path, model_path, "unet")
        print("\n" + "="*80)
        print("DIAGNOSIS COMPLETED")
        print("="*80)
        
    except Exception as e:
        print(f"Error in diagnosis: {e}")
        import traceback
        traceback.print_exc()