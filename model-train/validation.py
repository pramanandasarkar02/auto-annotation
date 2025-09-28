import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.spatial.distance import directed_hausdorff

from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from data_loading import get_data_list, create_dataloaders
from model import create_qct_segmentation_model

def calculate_metrics(pred, target):
    """Calculate all validation metrics."""
    # Ensure both tensors have the same shape
    if pred.shape != target.shape:
        print(f"Warning: Shape mismatch - pred: {pred.shape}, target: {target.shape}")
        # Resize target to match prediction if needed
        if pred.size > target.size:
            # Pad target
            pad_size = pred.size - target.size
            target_flat = np.pad(target.flatten(), (0, pad_size), mode='constant', constant_values=0)
            pred_flat = pred.flatten()
        else:
            # Crop prediction
            pred_flat = pred.flatten()[:target.size]
            target_flat = target.flatten()
    else:
        pred_flat = pred.flatten()
        target_flat = target.flatten()
    
    # Apply threshold to prediction
    pred_flat = (pred_flat > 0.5).astype(int)
    target_flat = target_flat.astype(int)
    
    # Basic metrics
    intersection = np.sum(pred_flat * target_flat)
    pred_sum = np.sum(pred_flat)
    target_sum = np.sum(target_flat)
    
    dice = (2 * intersection) / (pred_sum + target_sum) if (pred_sum + target_sum) > 0 else 0
    union = np.sum((pred_flat + target_flat) > 0)
    iou = intersection / union if union > 0 else 0
    
    # Classification metrics
    tp = np.sum((pred_flat == 1) & (target_flat == 1))
    tn = np.sum((pred_flat == 0) & (target_flat == 0))
    fp = np.sum((pred_flat == 1) & (target_flat == 0))
    fn = np.sum((pred_flat == 0) & (target_flat == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(target_flat, pred_flat, zero_division=0)
    recall = recall_score(target_flat, pred_flat, zero_division=0)
    f1 = f1_score(target_flat, pred_flat, zero_division=0)
    accuracy = accuracy_score(target_flat, pred_flat)
    
    # Hausdorff distance
    try:
        # Reshape back to original shape for coordinate extraction
        pred_reshaped = pred_flat.reshape(pred.shape)
        target_reshaped = target_flat[:pred.size].reshape(pred.shape)
        
        pred_coords = np.argwhere(pred_reshaped > 0.5)
        target_coords = np.argwhere(target_reshaped > 0.5)
        
        if len(pred_coords) > 0 and len(target_coords) > 0:
            hausdorff = max(directed_hausdorff(pred_coords, target_coords)[0],
                          directed_hausdorff(target_coords, pred_coords)[0])
        else:
            hausdorff = 0
    except Exception as e:
        print(f"Warning: Hausdorff calculation failed: {e}")
        hausdorff = 0
    
    # Volume metrics
    pred_volume = np.sum(pred_flat > 0.5)
    target_volume = np.sum(target_flat > 0.5)
    volume_similarity = (2 * intersection) / (pred_volume + target_volume) if (pred_volume + target_volume) > 0 else 0
    
    return {
        'dice_score': dice,
        'iou': iou,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'hausdorff_distance': hausdorff,
        'volume_similarity': volume_similarity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    }

def visualize_case(image, prediction, ground_truth, case_name):
    """Create side-by-side visualization."""
    os.makedirs('./validation_results', exist_ok=True)
    
    try:
        # Ensure all arrays have the same spatial dimensions for the middle slice
        img_shape = image.shape
        pred_shape = prediction.shape
        gt_shape = ground_truth.shape
        
        print(f"Shapes - Image: {img_shape}, Prediction: {pred_shape}, Ground Truth: {gt_shape}")
        
        # Use the minimum depth dimension
        min_depth = min(img_shape[-1], pred_shape[-1], gt_shape[-1])
        slice_idx = min_depth // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image[0, :, :, slice_idx], cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Ground truth overlay
        axes[1].imshow(image[0, :, :, slice_idx], cmap='gray', alpha=0.7)
        if gt_shape[-1] > slice_idx:
            axes[1].imshow(ground_truth[0, :, :, slice_idx], cmap='Reds', alpha=0.5)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction overlay
        axes[2].imshow(image[0, :, :, slice_idx], cmap='gray', alpha=0.7)
        if pred_shape[-1] > slice_idx:
            axes[2].imshow(prediction[0, :, :, slice_idx], cmap='Blues', alpha=0.5)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.suptitle(case_name)
        plt.tight_layout()
        plt.savefig(f'./validation_results/{case_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Visualization failed for {case_name}: {e}")
        plt.close()

def validation(images_dir: str, labels_dir: str, MODELS_DIR: str, number_of_images: int = 5):
    """
    Validate models on random images and calculate comprehensive metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_names = ["unet", "unetr", "swinunetr", "segresnet"]
    train_types = {
        "right_ct2": lambda x: "right_CT2" in x["image"],
        "left_ct2": lambda x: "left_CT2" in x["image"], 
        "right_ct3": lambda x: "right_CT3" in x["image"],
        "left_ct3": lambda x: "left_CT3" in x["image"],
        "ct2": lambda x: "CT2" in x["image"],
        "ct3": lambda x: "CT3" in x["image"],
        "left": lambda x: "left" in x["image"].lower(),
        "right": lambda x: "right" in x["image"].lower(),
        "combined": lambda x: True
    }
    
    # Get random validation data
    all_data = get_data_list(images_dir, labels_dir)
    print(f"Loaded {len(all_data)} data points.")
    random.seed(42)
    val_data = random.sample(all_data, min(number_of_images, len(all_data)))
    
    results = {}
    post_pred = AsDiscrete(threshold=0.5)
    post_label = AsDiscrete(threshold=0.5)
    
    # Test each model with progress bar
    model_pbar = tqdm(model_names, desc="Testing models")
    for model_name in model_pbar:
        model_pbar.set_description(f"Testing {model_name}")
        results[model_name] = {}
        
        # Progress bar for training types
        train_type_pbar = tqdm(train_types.items(), desc=f"{model_name} training types", leave=False)
        for train_type, filter_fn in train_type_pbar:
            train_type_pbar.set_description(f"{model_name} - {train_type}")
            
            model_path = f"{MODELS_DIR}/{model_name}/{train_type}_model.pth"
            
            if not os.path.exists(model_path):
                continue
                
            # Filter validation data for this training type
            filtered_data = [item for item in val_data if filter_fn(item)]
            if not filtered_data:
                continue
            
            try:
                # Load model
                model = create_qct_segmentation_model(model_name).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                
                # Create dataloader
                _, val_loader = create_dataloaders(filtered_data, batch_size=1, cache_rate=0.0)
                
                # Skip if no validation loader
                if val_loader is None:
                    print(f"Warning: No validation loader created for {model_name} - {train_type}")
                    continue
                
                # Calculate metrics with progress bar
                all_metrics = []
                batch_pbar = tqdm(enumerate(val_loader), desc=f"Processing {train_type}", 
                                total=len(val_loader), leave=False)
                
                with torch.no_grad():
                    for idx, batch in batch_pbar:
                        batch_pbar.set_description(f"Processing batch {idx+1}/{len(val_loader)}")
                        
                        inputs = batch["image"].to(device)
                        labels = batch["label"].to(device)
                        
                        # Predict
                        outputs = sliding_window_inference(inputs, (96, 96, 96), 1, model)
                        outputs = post_pred(outputs)
                        labels = post_label(labels)
                        
                        # Convert to numpy arrays 
                        outputs_np = outputs.cpu().numpy()
                        labels_np = labels.cpu().numpy()
                        
                        # Calculate metrics
                        try:
                            metrics = calculate_metrics(outputs_np, labels_np)
                            all_metrics.append(metrics)
                        except Exception as e:
                            print(f"Error calculating metrics for {train_type}, batch {idx}: {e}")
                            continue
                        
                        # Visualize first case
                        if idx == 0:
                            case_name = f"{model_name}_{train_type}_case{idx}"
                            try:
                                visualize_case(inputs.cpu().numpy(), outputs_np, 
                                             labels_np, case_name)
                            except Exception as e:
                                print(f"Visualization failed for {case_name}: {e}")
                
                # Average metrics
                if all_metrics:
                    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
                    results[model_name][train_type] = avg_metrics
                    tqdm.write(f"  {model_name} - {train_type}: Dice={avg_metrics['dice_score']:.3f}")
                    
            except Exception as e:
                print(f"Error processing {model_name} - {train_type}: {e}")
                continue
    
    # Create results tables
    create_tables(results, model_names)
    return results

def create_tables(results, model_names):
    """Create and save result tables."""
    os.makedirs('./validation_results', exist_ok=True)
    
    # Medical metrics table
    medical_metrics = ['dice_score', 'iou', 'sensitivity', 'specificity', 'hausdorff_distance', 'volume_similarity']
    medical_data = {}
    
    for metric in medical_metrics:
        medical_data[metric] = {}
        for model in model_names:
            if model in results and 'combined' in results[model]:
                medical_data[metric][model] = results[model]['combined'][metric]
            else:
                medical_data[metric][model] = 0.0
    
    medical_df = pd.DataFrame(medical_data).T
    print("\nMODEL SCORE TABLE")
    print(medical_df.round(4))
    medical_df.round(4).to_csv('./validation_results/medical_metrics.csv')
    
    # General metrics table  
    general_metrics = ['precision', 'recall', 'f1_score', 'accuracy']
    general_data = {}
    
    for metric in general_metrics:
        general_data[metric] = {}
        for model in model_names:
            if model in results and 'combined' in results[model]:
                general_data[metric][model] = results[model]['combined'][metric]
            else:
                general_data[metric][model] = 0.0
    
    general_df = pd.DataFrame(general_data).T
    print("\nGENERAL SCORE TABLE")
    print(general_df.round(4))
    general_df.round(4).to_csv('./validation_results/general_metrics.csv')
    
    # Femur part cross-validation (using best model)
    if medical_data and any(medical_data['dice_score'].values()):
        best_model = max(model_names, key=lambda m: medical_data['dice_score'].get(m, 0))
        femur_parts = ['left', 'right', 'combined']
        femur_data = {}
        
        for train_part in femur_parts:
            femur_data[train_part] = {}
            for test_part in femur_parts:
                if best_model in results and train_part in results[best_model]:
                    femur_data[train_part][test_part] = results[best_model][train_part]['dice_score']
                else:
                    femur_data[train_part][test_part] = 0.0
        
        femur_df = pd.DataFrame(femur_data)
        print(f"\nFEMUR PART SCORE TABLE (Best Model: {best_model})")
        print(femur_df.round(4))
        femur_df.round(4).to_csv('./validation_results/femur_cross_validation.csv')
    else:
        print("\nNo valid results found for femur cross-validation table")

# Usage
if __name__ == "__main__":
    root_dir = "/home/user/auto-annotation/auto-annotation/dataset/femur-bone/data"
    IMAGES_DIR = os.path.join(root_dir, "image")
    LABELS_DIR = os.path.join(root_dir, "label")
    MODELS_DIR = "/home/user/auto-annotation/auto-annotation/models"
    
    results = validation(IMAGES_DIR, LABELS_DIR, MODELS_DIR, number_of_images=5)
    print("\nValidation completed! Check './validation_results/' for outputs.")