from monai.networks.nets import VNet
import torch

def create_vnet3d(
    in_channels=1,
    out_channels=5,
    dropout_prob_down=0.5,
    dropout_prob_up=0.5,
    spatial_dims=3,
    device="cuda" if torch.cuda.is_available() else "cpu",
) -> VNet:
    model = VNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob_down=dropout_prob_down,
        dropout_prob_up=dropout_prob_up,
    )
    return model.to(device)
