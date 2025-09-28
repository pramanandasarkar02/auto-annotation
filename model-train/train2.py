import os
import json
import torch
import numpy as np
from tqdm import tqdm
from monai.losses import DiceCELoss, DiceLoss, FocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from data_loading import create_dataloaders, get_data_list
from model import create_qct_segmentation_model
import torch.nn.functional as F

def calculate_bone_focused_class_weights(data_loader, num_classes=5, device='cuda'):
    """
    Calculate extreme bone-focused class weights for maximum bone class optimization.
    """
    print("Calculating bone-focused class weights...")
    
    class_counts = torch.zeros(num_classes)
    total_voxels = 0
    
    # Count class occurrences with progress bar
    pbar = tqdm(data_loader, desc="Analyzing class distribution")
    for batch_data in pbar:
        labels = batch_data["label"]
        for class_idx in range(num_classes):
            class_counts[class_idx] += (labels == class_idx).sum().item()
        total_voxels += labels.numel()
        
        # Update progress bar with current stats
        pbar.set_postfix({
            'Total_voxels': f'{total_voxels:,}',
            'Bone_count': f'{int(class_counts[1]):,}',
            'WrongBone_count': f'{int(class_counts[2]):,}'
        })
    
    class_frequencies = class_counts / total_voxels
    
    # BONE-FOCUSED: Extreme weighting for minority bone classes
    weights = torch.ones(num_classes)
    weights[0] = 1.0   # Background - extremely penalized
    weights[1] = 200.0  # Bone - maximum priority
    weights[2] = 100.0  # Wrong Bone - maximum priority  
    weights[3] = 1.0    # Muscle - heavily penalized
    weights[4] = 10.0    # Extra Part - moderate weight
    
    print(f"\nClass counts: {class_counts}")
    print(f"Class frequencies: {class_frequencies}")
    print(f"BONE-FOCUSED weights: {weights}")
    
    return weights.to(device)

def create_bone_focused_loss(class_weights, num_classes=5, alpha=0.25):
    """
    Create bone-focused loss combination optimized for bone classes.
    """
    print("Creating BONE-FOCUSED loss function")
    
    # High gamma focal loss for hard bone examples
    focal_loss = FocalLoss(
        alpha=alpha,
        gamma=3.0,  # Higher gamma for harder examples
        weight=class_weights,
        to_onehot_y=True,
        use_softmax=True,
        include_background=True
    )
    
    # Dice loss with extreme bone weighting
    bone_weights = torch.zeros_like(class_weights)
    bone_weights[1] = 1.0  # Bone only
    bone_weights[2] = 1.0  # Wrong Bone only
    
    bone_dice_loss = DiceLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        weight=bone_weights[1:]  # Exclude background
    )
    
    # Overall dice loss
    dice_loss = DiceLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=True,
        weight=class_weights
    )
    
    def combined_loss_fn(pred, target):
        focal = focal_loss(pred, target)
        dice = dice_loss(pred, target) 
        bone_dice = bone_dice_loss(pred, target) * 15.0  # 15x boost for bone dice
        
        # Heavy emphasis on bone-specific dice
        total_loss = 0.15 * focal + 0.15 * dice + 0.7 * bone_dice
        return total_loss, {
            'focal': focal.item(),
            'dice': dice.item(), 
            'bone_dice': bone_dice.item()
        }
    
    return combined_loss_fn

def calculate_bone_focused_metrics(pred, target, num_classes=5):
    """
    Calculate comprehensive bone-focused metrics.
    """
    # Convert to one-hot for dice calculation
    pred_classes = torch.argmax(pred, dim=1)
    pred_onehot = F.one_hot(pred_classes, num_classes).permute(0, 4, 1, 2, 3).float()
    target_onehot = F.one_hot(target.squeeze(1).long(), num_classes).permute(0, 4, 1, 2, 3).float()
    
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
    dice_metric.reset()
    dice_metric(pred_onehot, target_onehot)
    class_dice = dice_metric.aggregate()
    
    # Bone-specific metrics
    bone_dice = class_dice[1] if len(class_dice) > 1 else torch.tensor(0.0)
    wrong_bone_dice = class_dice[2] if len(class_dice) > 2 else torch.tensor(0.0)
    bone_combined = (bone_dice + wrong_bone_dice) / 2
    
    # Additional bone metrics
    background_dice = class_dice[0] if len(class_dice) > 0 else torch.tensor(0.0)
    muscle_dice = class_dice[3] if len(class_dice) > 3 else torch.tensor(0.0)
    extra_dice = class_dice[4] if len(class_dice) > 4 else torch.tensor(0.0)
    
    return {
        'class_dice': class_dice,
        'bone_combined': bone_combined,
        'bone_dice': bone_dice,
        'wrong_bone_dice': wrong_bone_dice,
        'background_dice': background_dice,
        'muscle_dice': muscle_dice,
        'extra_dice': extra_dice,
        'foreground_mean': class_dice[1:].mean() if len(class_dice) > 1 else torch.tensor(0.0)
    }

def validate_model_predictions(model, val_loader, device, num_classes=5):
    """
    Check model prediction distribution during validation.
    """
    model.eval()
    class_predictions = torch.zeros(num_classes)
    total_predictions = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating predictions", leave=False)
        for val_data in pbar:
            val_inputs = val_data["image"].to(device)
            
            val_outputs = sliding_window_inference(
                val_inputs, 
                roi_size=(64, 64, 64), 
                sw_batch_size=1,
                predictor=model,
                overlap=0.75
            )
            
            # Get predicted classes
            pred_classes = torch.argmax(val_outputs, dim=1)
            
            # Count predictions for each class
            for class_idx in range(num_classes):
                class_predictions[class_idx] += (pred_classes == class_idx).sum().item()
            total_predictions += pred_classes.numel()
            
            # Update progress
            bone_pct = class_predictions[1] / total_predictions * 100 if total_predictions > 0 else 0
            wrong_bone_pct = class_predictions[2] / total_predictions * 100 if total_predictions > 0 else 0
            pbar.set_postfix({
                'Bone%': f'{bone_pct:.1f}',
                'WrongBone%': f'{wrong_bone_pct:.1f}'
            })
    
    prediction_percentages = class_predictions / total_predictions * 100
    
    print("\nModel prediction distribution:")
    class_names = ['Background', 'Bone', 'Wrong Bone', 'Muscle', 'Extra Part']
    for i, (name, pct) in enumerate(zip(class_names, prediction_percentages)):
        print(f"  {name}: {pct:.2f}%")
    
    return prediction_percentages

def train_model(images_dir: str, labels_dir: str, max_epochs: int = 200, 
                            learning_rate: float = 0.0001, device: str = None):
    """
    BONE-FOCUSED ONLY training function with comprehensive metrics and progress tracking.
    """
    
    model_name = "unet"
    num_classes = 5
    class_names = ['Background', 'Bone', 'Wrong Bone', 'Muscle', 'Extra Part']
    
    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model directory
    model_dir = "./models/unet"
    os.makedirs(model_dir, exist_ok=True)

    set_determinism(seed=42)
    
    model_path = "./models/unet/combined_model.pth"
    print(f"\nStarting BONE-FOCUSED training")
    print(f"Model will be saved to: {model_path}")

    # Get data
    print("\nPreparing data...")
    data_list = get_data_list(images_dir, labels_dir)
    
    if not data_list:
        print("No data found! Please check your data paths.")
        return None

    print(f"Found {len(data_list)} data samples")

    # Create data loaders
    train_loader, val_loader = create_dataloaders(data_list, batch_size=1, cache_rate=1.0)

    if train_loader is None or val_loader is None:
        print("Failed to create data loaders!")
        return None

    print(f"Training samples: {len(train_loader)}")
    print(f"Validation samples: {len(val_loader)}")

    # Calculate bone-focused class weights
    class_weights = calculate_bone_focused_class_weights(train_loader, num_classes, device)

    # Create model
    print("\nCreating model...")
    model = create_qct_segmentation_model(model_name).to(device)
    
    # Create bone-focused loss function
    loss_function = create_bone_focused_loss(class_weights, num_classes)
    
    # Optimizer with lower learning rate for stability
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
    )

    # Tracking variables
    val_interval = 5
    best_bone_metric = -1
    best_metric_epoch = -1
    patience_counter = 0
    early_stopping_patience = 50

    # Training history
    training_history = {
        'train_loss': [],
        'train_focal_loss': [],
        'train_dice_loss': [],
        'train_bone_dice_loss': [],
        'val_bone_combined': [],
        'val_bone_dice': [],
        'val_wrong_bone_dice': [],
        'val_foreground_mean': [],
        'learning_rates': []
    }

    print(f"\nStarting training for {max_epochs} epochs...")
    print("="*80)

    # Main training loop
    for epoch in range(max_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{max_epochs} | LR: {current_lr:.8f}")
        
        # Training phase
        model.train()
        epoch_losses = {'total': 0, 'focal': 0, 'dice': 0, 'bone_dice': 0}
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for step, batch_data in enumerate(train_pbar):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss, loss_components = loss_function(outputs, labels)
            
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update loss tracking
            epoch_losses['total'] += loss.item()
            epoch_losses['focal'] += loss_components['focal']
            epoch_losses['dice'] += loss_components['dice']
            epoch_losses['bone_dice'] += loss_components['bone_dice']

            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Bone_Dice_Loss': f'{loss_components["bone_dice"]:.4f}',
                'Focal': f'{loss_components["focal"]:.4f}'
            })

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        # Store training history
        training_history['train_loss'].append(epoch_losses['total'])
        training_history['train_focal_loss'].append(epoch_losses['focal'])
        training_history['train_dice_loss'].append(epoch_losses['dice'])
        training_history['train_bone_dice_loss'].append(epoch_losses['bone_dice'])
        training_history['learning_rates'].append(current_lr)

        print(f"Train Loss: {epoch_losses['total']:.4f} | "
              f"Focal: {epoch_losses['focal']:.4f} | "
              f"Dice: {epoch_losses['dice']:.4f} | "
              f"Bone_Dice: {epoch_losses['bone_dice']:.4f}")

        # Update scheduler
        scheduler.step()

        # Validation
        if (epoch + 1) % val_interval == 0:
            print(f"\nValidation at epoch {epoch+1}...")
            model.eval()
            
            # Check prediction distribution
            pred_dist = validate_model_predictions(model, val_loader, device, num_classes)
            
            with torch.no_grad():
                val_losses = {'total': 0, 'focal': 0, 'dice': 0, 'bone_dice': 0}
                all_metrics = []
                
                val_pbar = tqdm(val_loader, desc="Validation", leave=False)
                for val_data in val_pbar:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    
                    val_outputs = sliding_window_inference(
                        val_inputs, 
                        roi_size=(64, 64, 64),
                        sw_batch_size=1, 
                        predictor=model,
                        overlap=0.75
                    )
                    
                    # Calculate loss
                    val_loss, val_loss_components = loss_function(val_outputs, val_labels)
                    val_losses['total'] += val_loss.item()
                    val_losses['focal'] += val_loss_components['focal']
                    val_losses['dice'] += val_loss_components['dice']
                    val_losses['bone_dice'] += val_loss_components['bone_dice']
                    
                    # Calculate metrics
                    metrics = calculate_bone_focused_metrics(val_outputs, val_labels, num_classes)
                    all_metrics.append({k: v.cpu() if torch.is_tensor(v) else v for k, v in metrics.items()})
                    
                    # Update progress
                    val_pbar.set_postfix({
                        'Loss': f'{val_loss.item():.4f}',
                        'Bone_Combined': f'{metrics["bone_combined"]:.4f}'
                    })

                # Average validation losses
                for key in val_losses:
                    val_losses[key] /= len(val_loader)
                
                # Average validation metrics
                avg_metrics = {}
                for key in all_metrics[0].keys():
                    if key == 'class_dice':
                        avg_metrics[key] = torch.stack([m[key] for m in all_metrics]).mean(dim=0)
                    else:
                        avg_metrics[key] = torch.stack([m[key] for m in all_metrics]).mean()
                
                current_bone_metric = avg_metrics['bone_combined'].item()
                
                # Store validation history
                training_history['val_bone_combined'].append(current_bone_metric)
                training_history['val_bone_dice'].append(avg_metrics['bone_dice'].item())
                training_history['val_wrong_bone_dice'].append(avg_metrics['wrong_bone_dice'].item())
                training_history['val_foreground_mean'].append(avg_metrics['foreground_mean'].item())
                
                # Print detailed validation results
                print(f"\n{'='*60}")
                print(f"VALIDATION RESULTS - Epoch {epoch+1}")
                print(f"{'='*60}")
                print(f"Validation Losses:")
                print(f"  Total Loss: {val_losses['total']:.4f}")
                print(f"  Focal Loss: {val_losses['focal']:.4f}")
                print(f"  Dice Loss: {val_losses['dice']:.4f}")
                print(f"  Bone Dice Loss: {val_losses['bone_dice']:.4f}")
                
                print(f"\nBONE-FOCUSED METRICS:")
                print(f"  Combined Bone Dice: {current_bone_metric:.4f}")
                print(f"  Bone Dice: {avg_metrics['bone_dice']:.4f}")
                print(f"  Wrong Bone Dice: {avg_metrics['wrong_bone_dice']:.4f}")
                print(f"  Foreground Mean: {avg_metrics['foreground_mean']:.4f}")
                
                print(f"\nClass-wise Dice Scores:")
                for i, (class_name, dice_score) in enumerate(zip(class_names, avg_metrics['class_dice'])):
                    print(f"  {class_name} (Class {i}): {dice_score:.4f}")
                
                print(f"\nPrediction Distribution:")
                for i, (class_name, pct) in enumerate(zip(class_names, pred_dist)):
                    print(f"  {class_name}: {pct:.2f}%")
                
                # Check for improvement
                if current_bone_metric > best_bone_metric + 1e-6:
                    best_bone_metric = current_bone_metric
                    best_metric_epoch = epoch + 1
                    patience_counter = 0
                    
                    print(f"\nNEW BEST BONE METRIC: {best_bone_metric:.4f}")
                    
                    # Save best model
                    save_dict = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_bone_metric': best_bone_metric,
                        'class_dice_scores': avg_metrics['class_dice'].cpu().numpy(),
                        'prediction_distribution': pred_dist.numpy(),
                        'training_history': training_history,
                        'final_metrics': {k: v.item() if torch.is_tensor(v) else v 
                                        for k, v in avg_metrics.items() if k != 'class_dice'}
                    }
                    
                    torch.save(save_dict, model_path)
                    print(f"Saved new best model!")
                else:
                    patience_counter += 1
                    print(f"⏳ No improvement for {patience_counter} validation intervals")
                
                print(f"{'='*60}")
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {patience_counter} intervals without improvement")
                    break

    print(f"\nBONE-FOCUSED TRAINING COMPLETED!")
    print(f"Best Bone Metric: {best_bone_metric:.4f} at epoch {best_metric_epoch}")

    # Final results
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
            
            # Convert numpy arrays to lists for JSON serialization
            class_dice_scores = checkpoint.get('class_dice_scores', [])
            if hasattr(class_dice_scores, 'tolist'):
                class_dice_scores = class_dice_scores.tolist()
            elif isinstance(class_dice_scores, np.ndarray):
                class_dice_scores = class_dice_scores.tolist()
            
            prediction_distribution = checkpoint.get('prediction_distribution', [])
            if hasattr(prediction_distribution, 'tolist'):
                prediction_distribution = prediction_distribution.tolist()
            elif isinstance(prediction_distribution, np.ndarray):
                prediction_distribution = prediction_distribution.tolist()
            
            # Convert training history tensors to lists
            training_history_clean = {}
            if 'training_history' in checkpoint:
                for key, values in checkpoint['training_history'].items():
                    if isinstance(values, list):
                        # Convert any tensors in the list to float values
                        training_history_clean[key] = [
                            float(v.item()) if hasattr(v, 'item') else float(v) 
                            for v in values
                        ]
                    else:
                        training_history_clean[key] = values
            
            # Convert final metrics tensors to float values
            final_metrics_clean = {}
            if 'final_metrics' in checkpoint:
                for key, value in checkpoint['final_metrics'].items():
                    if hasattr(value, 'item'):
                        final_metrics_clean[key] = float(value.item())
                    elif isinstance(value, (torch.Tensor, np.ndarray)):
                        final_metrics_clean[key] = float(value)
                    else:
                        final_metrics_clean[key] = value
            
            final_results = {
                "training_type": "BONE-FOCUSED",
                "best_bone_metric": round(best_bone_metric, 4),
                "best_epoch": best_metric_epoch,
                "model_path": model_path,
                "final_class_dice_scores": {
                    class_names[i]: round(float(score), 4) 
                    for i, score in enumerate(class_dice_scores) if i < len(class_names)
                },
                "final_prediction_distribution": {
                    class_names[i]: round(float(pct), 2) 
                    for i, pct in enumerate(prediction_distribution) if i < len(class_names)
                },
                "final_metrics": final_metrics_clean,
                "training_history": training_history_clean
            }
            
            # Save detailed results
            results_path = "./training_results.json"
            with open(results_path, "w") as f:
                json.dump(final_results, f, indent=4)
            print(f"Detailed results saved to {results_path}")
            
            return final_results
            
        except Exception as e:
            print(f"Error loading final results from checkpoint: {e}")
            print("Returning basic results without checkpoint data...")
            
            # Fallback results without checkpoint data
            basic_results = {
                "training_type": "BONE-FOCUSED",
                "best_bone_metric": round(best_bone_metric, 4),
                "best_epoch": best_metric_epoch,
                "model_path": model_path,
                "error": f"Could not load full checkpoint data: {str(e)}",
                "note": "Model training completed but checkpoint loading failed"
            }
            
            # Save basic results
            results_path = "./bone_focused_training_results.json"
            with open(results_path, "w") as f:
                json.dump(basic_results, f, indent=4)
            print(f"Basic results saved to {results_path}")
            
            return basic_results
    
    return None



if __name__ == "__main__":
    # Configuration
    images_dir = "/path/to/images"
    labels_dir = "/path/to/labels"
    
    print("BONE-FOCUSED Multi-class Segmentation Training")
    print("="*80)
    print("This script focuses exclusively on optimizing bone class segmentation")
    print("="*80)
    
    # Run bone-focused training
    results = train_model(
        images_dir=images_dir,
        labels_dir=labels_dir,
        max_epochs=200,
        learning_rate=0.0001  # Lower learning rate for stability
    )
    
    if results:
        print("\n" + "="*80)
        print("BONE-FOCUSED TRAINING SUCCESSFULLY COMPLETED!")
        print("="*80)
        print(f"Best Bone Combined Dice: {results['best_bone_metric']}")
        print(f"Best Epoch: {results['best_epoch']}")
        print(f"Model Path: {results['model_path']}")
        print("\nFinal Class Dice Scores:")
        for class_name, score in results['final_class_dice_scores'].items():
            emoji = "🦴" if "Bone" in class_name else "📊"
            print(f"  {emoji} {class_name}: {score}")
    else:
        print("Training failed!")