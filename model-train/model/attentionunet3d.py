import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, UpSample
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """Custom Attention Gate implementation for 3D UNet"""
    def __init__(self, feature_channels, gating_channels, inter_channels):
        super().__init__()
        
        self.W_g = nn.Conv3d(gating_channels, inter_channels, kernel_size=1, bias=False)
        self.W_x = nn.Conv3d(feature_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, g):
        """
        Args:
            x: feature maps from encoder
            g: gating signal from coarser scale
        """
        # Apply transformations
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Add and apply activation
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.sigmoid(psi)
        
        # Apply attention
        return x * psi

class AttentionUNet3D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=5,
        feature_size=32,
        spatial_dims=3,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        # Encoder blocks
        self.encoder1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            strides=1,
            act=("relu", {"inplace": True}),
            norm="instance",
            bias=False,
        )
        self.encoder2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size * 2,
            strides=2,
            act=("relu", {"inplace": True}),
            norm="instance",
            bias=False,
        )
        self.encoder3 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size * 4,
            strides=2,
            act=("relu", {"inplace": True}),
            norm="instance",
            bias=False,
        )

        # Bottleneck
        self.bottleneck = Convolution(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 8,
            strides=2,
            act=("relu", {"inplace": True}),
            norm="instance",
            bias=False,
        )

        # Attention blocks + Decoder blocks
        self.att3 = AttentionGate(
            feature_channels=feature_size * 4,
            gating_channels=feature_size * 8,
            inter_channels=feature_size * 2,
        )
        self.decoder3 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            scale_factor=2,
        )
        self.conv_dec3 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            strides=1,
            act=("relu", {"inplace": True}),
            norm="instance",
            bias=False,
        )

        self.att2 = AttentionGate(
            feature_channels=feature_size * 2,
            gating_channels=feature_size * 4,
            inter_channels=feature_size,
        )
        self.decoder2 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            scale_factor=2,
        )
        self.conv_dec2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            strides=1,
            act=("relu", {"inplace": True}),
            norm="instance",
            bias=False,
        )

        self.att1 = AttentionGate(
            feature_channels=feature_size,
            gating_channels=feature_size * 2,
            inter_channels=feature_size // 2,
        )
        self.decoder1 = UpSample(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            scale_factor=2,
        )
        self.conv_dec1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            strides=1,
            act=("relu", {"inplace": True}),
            norm="instance",
            bias=False,
        )

        self.final_conv = nn.Conv3d(feature_size, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        b = self.bottleneck(e3)

        g3 = self.att3(e3, b)
        d3 = self.decoder3(b)
        d3 = torch.cat((d3, g3), dim=1)
        d3 = self.conv_dec3(d3)

        g2 = self.att2(e2, d3)
        d2 = self.decoder2(d3)
        d2 = torch.cat((d2, g2), dim=1)
        d2 = self.conv_dec2(d2)

        g1 = self.att1(e1, d2)
        d1 = self.decoder1(d2)
        d1 = torch.cat((d1, g1), dim=1)
        d1 = self.conv_dec1(d1)

        out = self.final_conv(d1)
        return out


def create_attention_unet3d(
    in_channels=1,
    out_channels=5,
    feature_size=32,
    spatial_dims=3,
    device="cuda" if torch.cuda.is_available() else "cpu",
) -> AttentionUNet3D:
    model = AttentionUNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        spatial_dims=spatial_dims,
    )
    return model.to(device)