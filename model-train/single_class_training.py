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



def create_loss_func():
    class_weights = torch.tensor([1.0, 150.0], dtype=torch.float32)

    focal_loss = FocalLoss(
        alpha=0.75,  # Focus more on positive class
        gamma=2.0,
        weight=class_weights,
        to_onehot_y=True,
        use_softmax=True,
        include_background=False  # Keep background for stable training
    )
    
    # Dice loss - IGNORE background, only compute on bone
    dice_loss = DiceLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,  # *** KEY: Only compute Dice on bone class ***
        squared_pred=True,
        smooth_nr=1e-5,
        smooth_dr=1e-5
    )
    
    def combined_loss_fn(pred, target):
        focal = focal_loss(pred, target)
        dice = dice_loss(pred, target)
        
        # Balanced combination
        total_loss = 0.5 * focal + 0.5 * dice
        return total_loss, {
            'focal': focal.item(),
            'dice': dice.item()
        }
    
    return combined_loss_fn


def train_model(images_dir, labels_dir, max_epochs = 100, learning_rate = 0.0001, device= None):
    num_classes = 2
    class_names = [ 'Background', 'Bone']

    model_dir = "./models"
    model_name = "unet_single_class.pth"
    
    model_path = os.path.join(model_dir, model_name)

    data_list = get_data_list(images_dir, labels_dir)
    train_loader, val_loader = create_dataloaders(data_list, batch_size=2, cache_rate=1.0)
    model = create_qct_segmentation_model("unet").to(device)

    loss_func = create_loss_func()

    # for epoch in range(max_epochs):
    #     current_lr = 