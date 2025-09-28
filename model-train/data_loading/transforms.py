# transforms.py
import os
from typing import Mapping, Hashable, Dict
import numpy as np

from monai.config import KeysCollection
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityd, CropForegroundd, RandCropByPosNegLabeld, ResizeWithPadOrCropd,
    ToTensord, EnsureTyped, SpatialPadd, MapTransform, RandFlipd, RandRotate90d, RandShiftIntensityd, RandScaleIntensityd, RandGaussianNoised, ScaleIntensityRanged
)

# def get_train_transforms():
#     """Get training transforms pipeline."""
#     return Compose([
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         Spacingd(
#             keys=["image", "label"],
#             pixdim=(1.0, 1.0, 1.0),
#             mode=("bilinear", "nearest")
#         ),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         ScaleIntensityd(
#             keys="image",
#             # a_min=-1000, a_max=1000,  # HU range for bone
#             # b_min=0.0, b_max=1.0, 
#             # clip=True
#         ),
#         CropForegroundd(keys=["image", "label"], source_key="image"),
#         SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
#         RandCropByPosNegLabeld(
#             keys=["image", "label"],
#             label_key="label",
#             spatial_size=(96, 96, 96),
#             pos=1, neg=1, num_samples=4,
#             image_key="image",
#             image_threshold=0,
#             # allow_smaller=True
#         ),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
#         RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
#         RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
#         EnsureTyped(keys=["image", "label"]),
#         ToTensord(keys=["image", "label"]),
#     ])

# def get_val_transforms():
#     """Get validation transforms pipeline."""
#     return Compose([
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         Spacingd(
#             keys=["image", "label"],
#             pixdim=(1.0, 1.0, 1.0),
#             mode=("bilinear", "nearest")
#         ),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         ScaleIntensityd(keys="image"),
#         CropForegroundd(keys=["image", "label"], source_key="image"),
#         EnsureTyped(keys=["image", "label"]),
#         ToTensord(keys=["image", "label"]),
#     ])
def get_train_transforms():
    """Get improved training transforms pipeline."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Improved intensity scaling for CT images
        ScaleIntensityRanged(
            keys="image",
            a_min=-1000, a_max=1000,  # HU range for bone/soft tissue
            b_min=0.0, b_max=1.0, 
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        # Ensure minimum size before cropping
        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
        
        # Improved random cropping strategy
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=2,  # More positive samples
            neg=1,  # Fewer negative samples
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        
        # Enhanced augmentations
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandGaussianNoised(keys=["image"], std=0.05, prob=0.3),
        
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])

def get_val_transforms():
    """Get validation transforms pipeline."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys="image",
            a_min=-1000, a_max=1000,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])