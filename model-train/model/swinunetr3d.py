import torch
from monai.networks.nets import SwinUNETR


def create_swin_unetr3d(
    in_channels: int = 1,
    out_channels: int = 5,
    feature_size: int = 24,  # use 24 for low-memory; 48 or 96 for higher capacity
    depths=(2, 2, 2, 2),
    num_heads=(3, 6, 12, 24),
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    dropout_path_rate: float = 0.0,
    use_checkpoint: bool = True,  # True saves memory at cost of speed
    norm_name: str = "instance",
    spatial_dims: int = 3,
    downsample: str = "merging",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> SwinUNETR:
    
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        depths=depths,
        num_heads=num_heads,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
        norm_name=norm_name,
        spatial_dims=spatial_dims,
        downsample=downsample,
    )
    return model
