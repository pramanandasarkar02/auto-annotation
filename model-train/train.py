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
# import torch.nn as nn


def create_loss_function():
    """
    Create combined loss function for segmentation using DiceCE and Dice loss.
    """
    print("Creating loss function")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor([.01, .99, .94, .09, 0.1], dtype=torch.float32).to(device)
    
    # DiceCE Loss (combines Dice and Cross Entropy)
    dice_ce_loss = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,  
        weight=class_weights
    )
    
    # Pure Dice Loss - minimal parameters to avoid version conflicts
    dice_loss = DiceLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        weight=class_weights
    )
    
    def combined_loss_fn(pred, target):
        dice_ce = dice_ce_loss(pred, target)
        dice = dice_loss(pred, target)
        
        # Combine losses with weights
        total_loss = dice_ce + 0.5 * dice
        
        return total_loss, {
            'dice_ce': dice_ce.item(),
            'dice': dice.item()
        }
    
    return combined_loss_fn


    
    

def calculate_metrics(pred, target, num_classes=5):
    
    # Convert to one-hot for dice calculation
    pred_classes = torch.argmax(pred, dim=1)
    pred_onehot = F.one_hot(pred_classes, num_classes).permute(0, 4, 1, 2, 3).float()
    target_onehot = F.one_hot(target.squeeze(1).long(), num_classes).permute(0, 4, 1, 2, 3).float()
    
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
    dice_metric.reset()
    dice_metric(pred_onehot, target_onehot)
    class_dice = dice_metric.aggregate()
    
    # Calculate mean dice for different class groups
    background_dice = class_dice[0] if len(class_dice) > 0 else torch.tensor(0.0)
    foreground_mean = class_dice[1:].mean() if len(class_dice) > 1 else torch.tensor(0.0)
    overall_mean = class_dice.mean()
    
    return {
        'class_dice': class_dice,
        'background_dice': background_dice,
        'foreground_mean': foreground_mean,
        'overall_mean': overall_mean
    }

def validate_predictions(model, val_loader, device, num_classes=5):
    
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
            pbar.set_postfix({
                'Total_predictions': f'{total_predictions:,}'
            })
    
    prediction_percentages = class_predictions / total_predictions * 100
    
    print("\nModel prediction distribution:")
    class_names = ['Background', 'Bone', 'Wrong Bone', 'Muscle', 'Extra Part']
    for i, (name, pct) in enumerate(zip(class_names, prediction_percentages)):
        print(f"  {name}: {pct:.2f}%")
    
    return prediction_percentages

def train_model(images_dir: str, labels_dir: str, max_epochs: int = 200, 
                learning_rate: float = 0.0001, device: str = None,
                model_names: list = None, train_types: dict = None):
    
    
    num_classes = 5
    class_names = ['Background', 'Bone', 'Wrong Bone', 'Muscle', 'Extra Part']
    # class_names = [ 'Bone']

    # Default model names if not provided
    if model_names is None:
        model_names = ["unet", "dynunet", "unetr", "swinunetr", "segresnet", "attnunet", "vnet"]
    model_names = ["swinunetr"]
    # model_names = ["csa"]
    # Default training types if not provided
    if train_types is None:
        train_types = {
            # "right_ct2": {
            #     "filter_fn": lambda x: "right_CT2" in x["image"],
            #     "description": "Training on right femur CT2 scans"
            # },
            # "left_ct2": {
            #     "filter_fn": lambda x: "left_CT2" in x["image"],
            #     "description": "Training on left femur CT2 scans"
            # },
            # "right_ct3": {
            #     "filter_fn": lambda x: "right_CT3" in x["image"],
            #     "description": "Training on right femur CT3 scans"
            # },
            # "left_ct3": {
            #     "filter_fn": lambda x: "left_CT3" in x["image"],
            #     "description": "Training on left femur CT3 scans"
            # },
            # "ct2": {
            #     "filter_fn": lambda x: "CT2" in x["image"],
            #     "description": "Training on all CT2 scans"
            # },
            # "ct3": {
            #     "filter_fn": lambda x: "CT3" in x["image"],
            #     "description": "Training on all CT3 scans"
            # },
            "left": {
                "filter_fn": lambda x: "left" in x["image"].lower(),
                "description": "Training on all left femur scans"
            },
            # "right": {
            #     "filter_fn": lambda x: "right" in x["image"].lower(),
            #     "description": "Training on all right femur scans"
            # },
            # "combined": {
            #     "filter_fn": lambda x: True,
            #     "description": "Training on all scans (CT2 and CT3, left and right)"
            # }
        }
    
    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create models directory
    origin_dir = "./models"
    os.makedirs(origin_dir, exist_ok=True)

    # Store all results
    all_results = {}

    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Training Model: {model_name.upper()}")
        print(f"{'='*80}")
        
        model_dir = os.path.join(origin_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        set_determinism(seed=42)
        
        model_results = {}

        for train_key, config in train_types.items():
            print(f"\nStarting training for {train_key}: {config['description']}")
            
            model_path = os.path.join(model_dir, f"{train_key}_model.pth")
            print(f"Model will be saved to: {model_path}")

            # Get filtered data list
            data_list = get_data_list(images_dir, labels_dir)
            filtered_data_list = [item for item in data_list if config["filter_fn"](item)]
            
            if not filtered_data_list:
                print(f"No data found for {train_key}. Skipping...")
                continue

            print(f"Found {len(filtered_data_list)} data samples")

            # Create data loaders
            train_loader, val_loader = create_dataloaders(filtered_data_list, batch_size=1, cache_rate=1.0)

            if train_loader is None or val_loader is None:
                print("Failed to create data loaders!")
                continue

            print(f"Training samples: {len(train_loader)}")
            print(f"Validation samples: {len(val_loader)}")

            

            # Create model
            print("\nCreating model...")
            model = create_qct_segmentation_model(model_name)
            model = model.to(device)  # Move model to GPU
            

            # Create loss function
            loss_function = create_loss_function()
            # loss_function =create_single_class_loss()
            


            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=1e-5,
                betas=(0.9, 0.999)
            )
            
            # Scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-7
            )

            # Tracking variables
            val_interval = 5
            best_metric = -1
            best_metric_epoch = -1
            patience_counter = 0
            early_stopping_patience = 10

            # Training history
            training_history = {
                'train_loss': [],
                'train_dice_ce_loss': [],
                'train_dice_loss': [],
                'val_overall_mean': [],
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
                epoch_losses = {'total': 0, 'dice_ce': 0, 'dice': 0}
                
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
                    epoch_losses['dice_ce'] += loss_components['dice_ce']
                    epoch_losses['dice'] += loss_components['dice']

                    # Update progress bar
                    train_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'DiceCE': f'{loss_components["dice_ce"]:.4f}',
                        'Dice': f'{loss_components["dice"]:.4f}'
                    })

                # Average losses
                for key in epoch_losses:
                    epoch_losses[key] /= len(train_loader)
                
                # Store training history
                training_history['train_loss'].append(epoch_losses['total'])
                training_history['train_dice_ce_loss'].append(epoch_losses['dice_ce'])
                training_history['train_dice_loss'].append(epoch_losses['dice'])
                training_history['learning_rates'].append(current_lr)

                print(f"Train Loss: {epoch_losses['total']:.4f} | "
                      f"DiceCE: {epoch_losses['dice_ce']:.4f} | "
                      f"Dice: {epoch_losses['dice']:.4f}")

                # Update scheduler
                scheduler.step()

                # Validation
                if (epoch + 1) % val_interval == 0:
                    print(f"\nValidation at epoch {epoch+1}...")
                    model.eval()
                    
                    # Check prediction distribution
                    pred_dist = validate_predictions(model, val_loader, device, num_classes)
                    
                    with torch.no_grad():
                        val_losses = {'total': 0, 'dice_ce': 0, 'dice': 0}
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
                            val_losses['dice_ce'] += val_loss_components['dice_ce']
                            val_losses['dice'] += val_loss_components['dice']
                            
                            # Calculate metrics
                            metrics = calculate_metrics(val_outputs, val_labels, num_classes)
                            all_metrics.append({k: v.cpu() if torch.is_tensor(v) else v for k, v in metrics.items()})
                            
                            # Update progress
                            val_pbar.set_postfix({
                                'Loss': f'{val_loss.item():.4f}',
                                'Overall_Mean': f'{metrics["overall_mean"]:.4f}'
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
                        
                        current_metric = avg_metrics['overall_mean'].item()
                        
                        # Store validation history
                        training_history['val_overall_mean'].append(current_metric)
                        training_history['val_foreground_mean'].append(avg_metrics['foreground_mean'].item())
                        
                        # Print detailed validation results
                        print(f"\n{'='*60}")
                        print(f"VALIDATION RESULTS - Epoch {epoch+1}")
                        print(f"{'='*60}")
                        print(f"Validation Losses:")
                        print(f"  Total Loss: {val_losses['total']:.4f}")
                        print(f"  DiceCE Loss: {val_losses['dice_ce']:.4f}")
                        print(f"  Dice Loss: {val_losses['dice']:.4f}")
                        
                        print(f"\nMETRICS:")
                        print(f"  Overall Mean Dice: {current_metric:.4f}")
                        print(f"  Foreground Mean Dice: {avg_metrics['foreground_mean']:.4f}")
                        print(f"  Background Dice: {avg_metrics['background_dice']:.4f}")
                        
                        print(f"\nClass-wise Dice Scores:")
                        for i, (class_name, dice_score) in enumerate(zip(class_names, avg_metrics['class_dice'])):
                            print(f"  {class_name} (Class {i}): {dice_score:.4f}")
                        
                        print(f"\nPrediction Distribution:")
                        for i, (class_name, pct) in enumerate(zip(class_names, pred_dist)):
                            print(f"  {class_name}: {pct:.2f}%")
                        
                        # Check for improvement
                        if current_metric > best_metric + 1e-6:
                            best_metric = current_metric
                            best_metric_epoch = epoch + 1
                            patience_counter = 0
                            
                            print(f"\nNEW BEST METRIC: {best_metric:.4f}")
                            
                            # Save best model
                            save_dict = {
                                'epoch': epoch + 1,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_metric': best_metric,
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
                            print(f"No improvement for {patience_counter} validation intervals")
                        
                        print(f"{'='*60}")
                        
                        # Early stopping
                        if patience_counter >= early_stopping_patience:
                            print(f"\nEarly stopping triggered after {patience_counter} intervals without improvement")
                            break

            print(f"\nTRAINING COMPLETED!")
            print(f"Best Metric: {best_metric:.4f} at epoch {best_metric_epoch}")

            # Final results processing
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
                    
                    results_data = {
                        "model_name": model_name,
                        "train_type": train_key,
                        "best_metric": round(best_metric, 4),
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
                    
                    model_results[train_key] = results_data
                    
                except Exception as e:
                    print(f"Error loading final results from checkpoint: {e}")
                    basic_results = {
                        "model_name": model_name,
                        "train_type": train_key,
                        "best_metric": round(best_metric, 4),
                        "best_epoch": best_metric_epoch,
                        "model_path": model_path,
                        "error": f"Could not load full checkpoint data: {str(e)}"
                    }
                    model_results[train_key] = basic_results

        all_results[model_name] = model_results
        
        # Save model-specific results
        model_results_path = os.path.join(model_dir, "training_results.json")
        with open(model_results_path, 'w') as f:
            json.dump(model_results, f, indent=4)
        print(f"Model {model_name} results saved to: {model_results_path}")

    # Save overall results
    overall_results_path = "./overall_training_results.json"
    with open(overall_results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"Overall results saved to: {overall_results_path}")
    
    return all_results

if __name__ == "__main__":
    # Configuration
    images_dir = "/path/to/images"
    labels_dir = "/path/to/labels"
    
    print("Multi-Model Segmentation Training")
    print("="*80)
    
    # Run training
    results = train_model(
        images_dir=images_dir,
        labels_dir=labels_dir,
        max_epochs=20,
        learning_rate=0.0001
    )
    
    if results:
        print("\nTraining completed successfully!")
        print("Check the './models/' directory for saved models and results.")
    else:
        print("Training failed!")