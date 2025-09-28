import os
from monai.data import DataLoader, CacheDataset, partition_dataset
from monai.data import pad_list_data_collate
from data_loading import get_data_list
from data_loading import get_train_transforms, get_val_transforms
import logging

def create_dataloaders(images_dir: str, labels_dir: str, batch_size: int = 2, cache_rate: float = 1.0):
    """
    Create training and validation dataloaders.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        batch_size: Batch size for training
        cache_rate: Cache rate for CacheDataset
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get data list
    data_list = get_data_list(images_dir, labels_dir)
    
    if not data_list:
        raise ValueError("No data found. Check your directory paths.")
    
    # Split data
    train_files, val_files = partition_dataset(
        data=data_list, 
        ratios=[0.8, 0.2], 
        shuffle=True
    )
    
    # Get transforms
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()
    
    # Create datasets
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
    
    print(f"Number of training samples: {len(train_ds)}")
    print(f"Number of validation samples: {len(val_ds)}")
    
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
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    return train_loader, val_loader



# def create_dataloaders(data_list: list, batch_size: int = 2, cache_rate: float = 1.0):
#     """
#     Create training and validation dataloaders from a pre-filtered data list.
    
#     Args:
#         data_list: List of dictionaries with image and label paths
#         batch_size: Batch size for training
#         cache_rate: Cache rate for CacheDataset
    
#     Returns:
#         Tuple of (train_loader, val_loader)
#     """
#     logger = logging.getLogger(__name__)
#     if not data_list:
#         raise ValueError("No data found in provided data list.")
    
#     # Split data into training and validation sets
#     train_files, val_files = partition_dataset(
#         data=data_list, 
#         ratios=[0.8, 0.2], 
#         shuffle=True
#     )
    
#     # Get transforms
#     train_transforms = get_train_transforms()
#     val_transforms = get_val_transforms()

    
#     # Create datasets
#     train_ds = CacheDataset(
#         data=train_files, 
#         transform=train_transforms, 
#         cache_rate=cache_rate, 
#         num_workers=4
#     )
#     val_ds = CacheDataset(
#         data=val_files, 
#         transform=val_transforms, 
#         cache_rate=cache_rate, 
#         num_workers=4
#     )
    
#     print(f"Number of training samples: {len(train_ds)}")
#     print(f"Number of validation samples: {len(val_ds)}")

#     if len(train_ds) == 0 or len(val_ds) == 0:
#         return None ,None
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_ds, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         num_workers=4, 
#         pin_memory=True, 
#         collate_fn=pad_list_data_collate
#     )
#     val_loader = DataLoader(
#         val_ds, 
#         batch_size=1, 
#         shuffle=False, 
#         num_workers=2, 
#         pin_memory=True
#     )
    
#     print(f"Number of training batches: {len(train_loader)}")
#     print(f"Number of validation batches: {len(val_loader)}")
    
#     return train_loader, val_loader

def create_dataloaders(data_list: list, batch_size: int = 2, cache_rate: float = 1.0, augment_copies: int = 1):
    """
    Create training and validation dataloaders with data augmentation via repeated transforms.

    Args:
        data_list: List of dicts with image and label paths.
        batch_size: Batch size for training.
        cache_rate: Cache rate for CacheDataset.
        augment_copies: Number of times to replicate each training sample with different transforms.

    Returns:
        Tuple of (train_loader, val_loader)
    """
    import logging
    from monai.data import CacheDataset, DataLoader, pad_list_data_collate
    from monai.transforms import Compose
    from monai.data.utils import partition_dataset

    # logger = logging.getLogger(__name__)
    if not data_list:
        raise ValueError("No data found in provided data list.")
    
    # Split data
    train_files, val_files = partition_dataset(
        data=data_list, 
        ratios=[0.7, 0.3], 
        shuffle=True
    )

    # Duplicate training files
    train_files = train_files * augment_copies
    val_files = val_files

    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

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

    print(f"Number of training samples: {len(train_ds)}")
    print(f"Number of validation samples: {len(val_ds)}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        return None, None

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

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    return train_loader, val_loader
