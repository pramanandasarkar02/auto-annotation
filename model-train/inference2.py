import os
import json
import torch
import numpy as np
from monai.losses import DiceCELoss, DiceLoss, FocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from data_loading import create_dataloaders, get_data_list
from model import create_qct_segmentation_model


class CombinedLoss(torch.nn.Module):
    """Combined loss function for better multi-class learning"""
    def __init__(self, class_weights=None, num_classes=5):
        super().__init__()
        self.dice_ce = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=True,
            weight=class_weights,
            lambda_dice=0.6,
            lambda_ce=0.4,
            smooth_nr=1e-5,
            smooth_dr=1e-5
        )
        
        # Focal loss to handle class imbalance
        self.focal = FocalLoss(
            to_onehot_y=True,
            weight=class_weights,
            gamma=2.0,
            alpha=0.25
        )
        
        # Individual class dice losses for minority classes
        self.class_dice_losses = torch.nn.ModuleList([
            DiceLoss(
                to_onehot_y=True,
                softmax=True,
                include_background=False,
                smooth_nr=1e-5,
                smooth_dr=1e-5
            ) for _ in range(num_classes-1)  # Exclude background
        ])
    
    def forward(self, pred, target):
        # Main combined loss
        main_loss = self.dice_ce(pred, target)
        
        # Focal loss for hard examples
        focal_loss = self.focal(pred, target)
        
        # Additional penalty for minority classes
        class_penalty = 0
        pred_softmax = torch.softmax(pred, dim=1)
        target_onehot = torch.zeros_like(pred_softmax)
        target_onehot.scatter_(1, target.long(), 1)
        
        # Focus on classes 2, 3, 4 which are underrepresented
        minority_classes = [2, 3, 4]
        for class_idx in minority_classes:
            if target_onehot[:, class_idx].sum() > 0:  # Only if class is present
                class_dice = self.class_dice_losses[class_idx-1](pred, target)
                class_penalty += class_dice * 2.0  # Extra weight for minority classes
        
        total_loss = main_loss + 0.3 * focal_loss + 0.2 * class_penalty
        return total_loss


def calculate_enhanced_class_weights(train_loader, num_classes=5, device='cuda'):
    """Calculate enhanced class weights with better balancing"""
    class_counts = torch.zeros(num_classes)
    
    print("Calculating enhanced class weights...")
    for batch_data in train_loader:
        labels = batch_data["label"]
        if labels.shape[1] > 1:
            labels = torch.argmax(labels, dim=1, keepdim=True)
        
        for class_id in range(num_classes):
            class_counts[class_id] += (labels == class_id).sum().item()
    
    # More aggressive weighting for minority classes
    total_pixels = class_counts.sum()
    class_frequencies = class_counts / total_pixels
    
    # Use effective number of samples weighting
    beta = 0.9999
    effective_nums = 1.0 - torch.pow(beta, class_counts)
    class_weights = (1.0 - beta) / effective_nums
    
    # Additional boost for very rare classes
    for i, freq in enumerate(class_frequencies):
        if freq < 0.05:  # Less than 5% of data
            class_weights[i] *= 3.0
        elif freq < 0.10:  # Less than 10% of data
            class_weights[i] *= 2.0
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"Class counts: {class_counts}")
    print(f"Class frequencies: {class_frequencies}")
    print(f"Enhanced class weights: {class_weights}")
    
    return class_weights.to(device)


def analyze_detailed_predictions(outputs, labels, epoch, step):
    """More detailed analysis of predictions"""
    with torch.no_grad():
        pred_probs = torch.softmax(outputs, dim=1)
        pred_classes = torch.argmax(outputs, dim=1)
        
        # Check confidence for each class
        for class_id in range(5):
            class_mask = (labels == class_id)
            if class_mask.sum() > 0:
                class_confidence = pred_probs[:, class_id][class_mask].mean()
                predicted_correctly = (pred_classes == class_id)[class_mask].float().mean()
                
                if step % 20 == 0 and class_id > 1:  # Focus on minority classes
                    print(f"    Class {class_id}: Avg confidence: {class_confidence:.4f}, "
                          f"Accuracy: {predicted_correctly:.4f}")


def train_model(images_dir: str, labels_dir: str, max_epochs: int = 50, 
                learning_rate: float = 0.0005, device: str = None):
    """Enhanced training function for better multi-class learning"""
    
    model_names = ["unet"]
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    origin_dir = "./models"
    os.makedirs(origin_dir, exist_ok=True)
    training_results = {}

    for model_name in model_names:
        model_dir = os.path.join(origin_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        set_determinism(seed=42)
        
        train_type = {
            "combined": {
                "filter_fn": lambda x: True,
                "model_path": f"./models/{model_name}/combined_model.pth",
                "description": "Training on all scans with enhanced multi-class learning"
            }
        }

        training_results[model_name] = {}

        for train_key, config in train_type.items():
            print(f"\nStarting enhanced training for {train_key}: {config['description']}")

            data_list = get_data_list(images_dir, labels_dir)
            filtered_data_list = [item for item in data_list if config["filter_fn"](item)]

            if not filtered_data_list:
                print(f"No data found for {train_key}. Skipping...")
                continue

            print(f"Found {len(filtered_data_list)} files for training")

            train_loader, val_loader = create_dataloaders(
                filtered_data_list, 
                batch_size=1,
                cache_rate=0.5
            )

            if train_loader is None or val_loader is None:
                continue

            # Create model with dropout for better generalization
            model = create_qct_segmentation_model(model_name).to(device)
            
            # Enhanced class weights
            class_weights = calculate_enhanced_class_weights(train_loader, device=device)
            
            # Use combined loss function
            loss_function = CombinedLoss(class_weights=class_weights, num_classes=5)
            
            # More conservative optimizer with different parameters for different layers
            param_groups = [
                {'params': [p for name, p in model.named_parameters() if 'final' in name or 'out' in name], 
                 'lr': learning_rate, 'weight_decay': 1e-5},  # Lower regularization for final layers
                {'params': [p for name, p in model.named_parameters() if 'final' not in name and 'out' not in name], 
                 'lr': learning_rate * 0.1, 'weight_decay': 1e-4}  # Lower LR for backbone
            ]
            
            optimizer = torch.optim.AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))
            
            # Cosine annealing with warm restarts
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-7
            )
            
            # Metrics for each class
            dice_metrics_per_class = [
                DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False),
                DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
            ]
            
            post_pred = AsDiscrete(argmax=True, to_onehot=5)
            post_label = AsDiscrete(to_onehot=5)

            val_interval = 2
            best_metric = -1
            best_metric_epoch = -1
            patience_counter = 0
            early_stopping_patience = 20
            
            # Track per-class performance
            best_class_metrics = torch.zeros(5)

            for epoch in range(max_epochs):
                print(f"\nEpoch {epoch+1}/{max_epochs}")
                print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
                
                model.train()
                epoch_loss = 0
                step = 0

                for batch_data in train_loader:
                    step += 1
                    inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                    
                    if labels.shape[1] > 1:
                        labels = torch.argmax(labels, dim=1, keepdim=True)
                    labels = labels.long()
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    
                    # Detailed prediction analysis for first few epochs
                    if epoch < 10:
                        analyze_detailed_predictions(outputs, labels, epoch, step)
                    
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    epoch_loss += loss.item()

                    if step % 10 == 0:
                        print(f"[Train] Step {step}/{len(train_loader)}, Loss: {loss.item():.4f}")

                epoch_loss /= step
                scheduler.step()  # Update learning rate
                print(f"Epoch {epoch+1} average loss: {epoch_loss:.4f}")

                # Validation with detailed per-class analysis
                if (epoch + 1) % val_interval == 0:
                    model.eval()
                    val_loss = 0
                    val_steps = 0
                    
                    # Reset metrics
                    for metric in dice_metrics_per_class:
                        metric.reset()
                    
                    class_intersection = torch.zeros(5)
                    class_union = torch.zeros(5)
                    
                    with torch.no_grad():
                        for val_data in val_loader:
                            val_steps += 1
                            val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                            
                            if val_labels.shape[1] > 1:
                                val_labels = torch.argmax(val_labels, dim=1, keepdim=True)
                            val_labels = val_labels.long()
                            
                            # Multi-scale inference for better results
                            scales = [(96, 96, 96), (128, 128, 128)]
                            val_outputs = None
                            
                            for roi_size in scales:
                                scale_output = sliding_window_inference(
                                    val_inputs, 
                                    roi_size=roi_size,
                                    sw_batch_size=2, 
                                    predictor=model,
                                    overlap=0.6
                                )
                                if val_outputs is None:
                                    val_outputs = scale_output
                                else:
                                    val_outputs += scale_output
                            
                            val_outputs /= len(scales)  # Average predictions
                            val_loss += loss_function(val_outputs, val_labels).item()
                            
                            # Calculate per-class IoU manually
                            pred_classes = torch.argmax(val_outputs, dim=1)
                            for class_id in range(5):
                                pred_mask = (pred_classes == class_id)
                                true_mask = (val_labels.squeeze(1) == class_id)
                                
                                intersection = (pred_mask & true_mask).sum().float()
                                union = (pred_mask | true_mask).sum().float()
                                
                                class_intersection[class_id] += intersection
                                class_union[class_id] += union
                            
                            # MONAI metrics
                            val_outputs_discrete = post_pred(val_outputs)
                            val_labels_discrete = post_label(val_labels)
                            
                            dice_metrics_per_class[0](y_pred=val_outputs_discrete, y=val_labels_discrete)
                            dice_metrics_per_class[1](y_pred=val_outputs_discrete, y=val_labels_discrete)

                        # Calculate IoU per class
                        class_ious = []
                        for class_id in range(5):
                            if class_union[class_id] > 0:
                                iou = class_intersection[class_id] / class_union[class_id]
                                class_ious.append(iou.item())
                            else:
                                class_ious.append(0.0)
                        
                        # MONAI Dice scores
                        try:
                            dice_with_bg = dice_metrics_per_class[0].aggregate()
                            dice_no_bg = dice_metrics_per_class[1].aggregate()
                            
                            if dice_with_bg.numel() > 1:
                                per_class_dice_bg = dice_with_bg.cpu().numpy()
                                mean_dice_bg = dice_with_bg.mean().item()
                            else:
                                per_class_dice_bg = [dice_with_bg.item()]
                                mean_dice_bg = dice_with_bg.item()
                                
                            if dice_no_bg.numel() > 1:
                                per_class_dice_fg = dice_no_bg.cpu().numpy()
                                mean_dice_fg = dice_no_bg.mean().item()
                            else:
                                per_class_dice_fg = [dice_no_bg.item()]
                                mean_dice_fg = dice_no_bg.item()
                            
                            val_loss /= val_steps
                            
                            # Composite metric: average of foreground dice and minority class performance
                            minority_classes_dice = np.mean([per_class_dice_bg[i] for i in [2, 3, 4] if i < len(per_class_dice_bg)])
                            primary_metric = 0.7 * mean_dice_fg + 0.3 * minority_classes_dice
                            
                        except Exception as e:
                            print(f"Error calculating metrics: {e}")
                            primary_metric = 0
                            per_class_dice_bg = []
                            per_class_dice_fg = []
                            mean_dice_bg = 0
                            mean_dice_fg = 0
                            minority_classes_dice = 0
                        
                        print(f"[Val] Epoch {epoch+1}")
                        print(f"  Validation Loss: {val_loss:.4f}")
                        print(f"  Overall Dice (with bg): {mean_dice_bg:.4f}")
                        print(f"  Overall Dice (no bg): {mean_dice_fg:.4f}")
                        print(f"  Minority classes Dice: {minority_classes_dice:.4f}")
                        print(f"  Per-class Dice: {per_class_dice_bg}")
                        print(f"  Per-class IoU: {[f'{iou:.4f}' for iou in class_ious]}")
                        print(f"  Composite metric: {primary_metric:.4f}")
                        
                        # Save best model
                        if primary_metric > best_metric:
                            best_metric = primary_metric
                            best_metric_epoch = epoch + 1
                            patience_counter = 0
                            best_class_metrics = torch.tensor(per_class_dice_bg[:5] if len(per_class_dice_bg) >= 5 else per_class_dice_bg + [0]*(5-len(per_class_dice_bg)))
                            
                            torch.save({
                                'epoch': epoch + 1,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_metric': best_metric,
                                'model_name': model_name,
                                'per_class_dice': per_class_dice_bg,
                                'per_class_iou': class_ious,
                                'class_weights': class_weights.cpu()
                            }, config["model_path"])
                            
                            print(f"✓ Saved new best model!")
                            print(f"  Best composite metric: {best_metric:.4f} at epoch {best_metric_epoch}")
                        else:
                            patience_counter += 1
                            print(f"  No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                        
                        # Early stopping
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping triggered after {epoch+1} epochs")
                            break
                    
                    # Reset metrics
                    for metric in dice_metrics_per_class:
                        metric.reset()

            # Save results
            training_results[model_name][train_key] = {
                "description": config["description"],
                "best_composite_metric": round(best_metric, 4),
                "best_epoch": best_metric_epoch,
                "model_path": config["model_path"],
                "best_per_class_dice": [round(x, 4) for x in best_class_metrics.tolist()]
            }

    # Save results to JSON
    results_path = "./enhanced-model-train.json"
    with open(results_path, "w") as f:
        json.dump(training_results, f, indent=4)
    print(f"\nEnhanced training results saved to {results_path}")

    return training_results


# Additional utility function for data augmentation during training
def get_enhanced_transforms():
    """Get enhanced transforms that help with minority class learning"""
    from monai.transforms import (
        Compose, RandRotated, RandZoomd, RandGaussianNoised, 
        RandAdjustContrastd, RandShiftIntensityd, RandFlipd,
        RandCropByPosNegLabeld
    )
    
    train_transforms = Compose([
        # Existing transforms...
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=2,  # Higher probability of including foreground
            neg=1,
            num_samples=2,  # Generate multiple crops per image
            image_key="image",
            image_threshold=0,
        ),
        RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.3),
        RandZoomd(keys=["image", "label"], min_zoom=0.8, max_zoom=1.2, prob=0.3),
        RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.3),
        RandGaussianNoised(keys="image", std=0.01, prob=0.2),
        RandAdjustContrastd(keys="image", gamma=(0.8, 1.2), prob=0.2),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
    ])
    
    return train_transforms