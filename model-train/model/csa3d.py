import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_cat = self.conv1(x_cat)
        return self.sigmoid(x_cat)


class CSABlock(nn.Module):
    """Channel and Spatial Attention Block"""
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super(CSABlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        x = x * self.ca(x)
        # Spatial attention
        x = x * self.sa(x)
        return x


class ConvBlock(nn.Module):
    """Basic Convolutional Block with CSA"""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        use_csa: bool = True,
        norm_name: str = "instance",
        dropout_rate: float = 0.0
    ):
        super(ConvBlock, self).__init__()
        
        # Normalization layer
        if norm_name.lower() == "batch":
            norm_layer = nn.BatchNorm3d
        elif norm_name.lower() == "instance":
            norm_layer = nn.InstanceNorm3d
        else:
            norm_layer = nn.Identity
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = norm_layer(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = norm_layer(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # CSA block
        self.csa = CSABlock(out_channels) if use_csa else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.dropout(x)
        x = self.csa(x)
        return x


class DownBlock(nn.Module):
    """Downsampling Block"""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        use_csa: bool = True,
        norm_name: str = "instance",
        dropout_rate: float = 0.0
    ):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_channels, out_channels, use_csa, norm_name, dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """Upsampling Block"""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        use_csa: bool = True,
        norm_name: str = "instance",
        dropout_rate: float = 0.0
    ):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, use_csa, norm_name, dropout_rate)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size mismatch
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2,
                        diff_z // 2, diff_z - diff_z // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class CSANetwork(nn.Module):
    """
    CSA Network for 3D Image Segmentation
    
    A U-Net-like architecture enhanced with Channel and Spatial Attention mechanisms.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 5,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512),
        use_csa: bool = True,
        norm_name: str = "instance",
        dropout_rate: float = 0.0,
        deep_supervision: bool = False
    ):
        super(CSANetwork, self).__init__()
        
        self.features = features
        self.use_csa = use_csa
        self.deep_supervision = deep_supervision
        
        # Input convolution
        self.input_conv = ConvBlock(
            in_channels, features[0], use_csa, norm_name, dropout_rate
        )
        
        # Encoder (downsampling path)
        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(
                DownBlock(features[i], features[i + 1], use_csa, norm_name, dropout_rate)
            )
        
        # Decoder (upsampling path)
        self.ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(
                UpBlock(features[i], features[i - 1], use_csa, norm_name, dropout_rate)
            )
        
        # Output convolution
        self.output_conv = nn.Conv3d(features[0], out_channels, 1)
        
        # Deep supervision outputs (optional)
        if deep_supervision:
            self.deep_outputs = nn.ModuleList([
                nn.Conv3d(features[i], out_channels, 1) 
                for i in range(len(features) - 1)
            ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store skip connections
        skip_connections = []
        
        # Input
        x = self.input_conv(x)
        skip_connections.append(x)
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        
        # Remove the last skip connection (it's the bottleneck)
        skip_connections = skip_connections[:-1]
        
        # Decoder
        deep_outputs = []
        for i, up in enumerate(self.ups):
            x = up(x, skip_connections[-(i + 1)])
            if self.deep_supervision and i < len(self.ups) - 1:
                deep_out = self.deep_outputs[i](x)
                deep_outputs.append(deep_out)
        
        # Final output
        output = self.output_conv(x)
        
        if self.deep_supervision and self.training:
            return [output] + deep_outputs
        else:
            return output


def create_csa_network(
    in_channels: int = 1,
    out_channels: int = 5,
    features: Tuple[int, ...] = (32, 64, 128, 256, 512),
    use_csa: bool = True,
    norm_name: str = "instance",
    dropout_rate: float = 0.0,
    deep_supervision: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> CSANetwork:
    """
    Create CSA Network for image segmentation
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        features: Feature sizes for each level
        use_csa: Whether to use Channel and Spatial Attention
        norm_name: Normalization type ("batch", "instance")
        dropout_rate: Dropout rate
        deep_supervision: Whether to use deep supervision
        device: Device to place the model on
    
    Returns:
        CSANetwork model
    """
    model = CSANetwork(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        use_csa=use_csa,
        norm_name=norm_name,
        dropout_rate=dropout_rate,
        deep_supervision=deep_supervision
    )
    
    return model.to(device)


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_csa_network(
        in_channels=1,
        out_channels=5,
        features=(32, 64, 128, 256, 512),
        use_csa=True,
        dropout_rate=0.1,
        deep_supervision=True
    )
    
    # Test with random input
    x = torch.randn(1, 1, 96, 96, 96)  # Batch, Channel, Depth, Height, Width
    
    with torch.no_grad():
        output = model(x)
        if isinstance(output, list):
            print(f"Main output shape: {output[0].shape}")
            print(f"Deep supervision outputs: {len(output)-1}")
        else:
            print(f"Output shape: {output.shape}")
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")