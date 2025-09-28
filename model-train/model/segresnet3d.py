import torch
from monai.networks.nets import (
    SegResNet
)


def create_segresnet3d(
    in_channels: int = 1,
    out_channels: int = 4,
    init_filters: int = 32,
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    dropout_prob: float = 0.0,
):
    return SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        dropout_prob=dropout_prob,
        norm="INSTANCE",
    )
