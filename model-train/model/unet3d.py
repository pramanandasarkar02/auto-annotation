import torch
from monai.networks.nets import UNet

def create_unet3d(in_channels: int = 1, out_channels: int = 5):
    """
    Create a 3D UNet model for medical image segmentation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (classes)
    
    Returns:
        UNet model
    """
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.3
    )
