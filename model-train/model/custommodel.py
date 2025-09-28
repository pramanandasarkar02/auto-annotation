import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math


class SpatialAttention(nn.Module):
    """Spatial attention module for bone structure focus"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module for feature enhancement"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class DenseASPP(nn.Module):
    """Dense ASPP for multi-scale feature extraction"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Reduce intermediate channels to prevent dimension explosion
        inter_channels = out_channels // 4
        
        self.conv1x1 = nn.Conv3d(in_channels, inter_channels, 1)
        self.conv3x3_1 = nn.Conv3d(in_channels, inter_channels, 3, padding=1, dilation=1)
        self.conv3x3_2 = nn.Conv3d(in_channels, inter_channels, 3, padding=2, dilation=2)
        self.conv3x3_3 = nn.Conv3d(in_channels, inter_channels, 3, padding=4, dilation=4)
        self.conv3x3_4 = nn.Conv3d(in_channels, inter_channels, 3, padding=8, dilation=8)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_conv = nn.Conv3d(in_channels, inter_channels, 1)
        
        # Calculate correct number of input channels for final conv
        # 6 branches: conv1x1, conv3x3_1-4, global_conv
        final_in_channels = 6 * inter_channels
        self.final_conv = nn.Conv3d(final_in_channels, out_channels, 1)
        
    def forward(self, x):
        size = x.shape[2:]
        
        # All branches process input independently (no concatenation in intermediate steps)
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_1(x)
        feat3 = self.conv3x3_2(x)
        feat4 = self.conv3x3_3(x)
        feat5 = self.conv3x3_4(x)
        
        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = self.global_conv(global_feat)
        global_feat = F.interpolate(global_feat, size=size, mode='trilinear', align_corners=False)
        
        # Concatenate all features
        final_feat = torch.cat([feat1, feat2, feat3, feat4, feat5, global_feat], dim=1)
        output = self.final_conv(final_feat)
        
        return output


class ResidualBlock(nn.Module):
    """Residual block with bone-specific modifications"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
            
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention(out_channels)
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        
        # Apply attention mechanisms
        out = self.channel_attention(out)
        out = self.spatial_attention(out)
        
        out += residual
        out = F.relu(out, inplace=True)
        
        return out


class BoneSegmentationHead(nn.Module):
    """Specialized segmentation head for bone structures"""
    def __init__(self, in_channels: int, num_classes: int, use_deep_supervision: bool = True):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision
        
        # Main segmentation head
        self.main_head = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(in_channels // 2, num_classes, 1)
        )
        
        # Deep supervision heads
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv3d(in_channels, num_classes, 1)
                for _ in range(3)
            ])
    
    def forward(self, features):
        main_output = self.main_head(features[-1])
        
        if self.use_deep_supervision and self.training:
            aux_outputs = []
            for i, aux_head in enumerate(self.aux_heads):
                aux_out = aux_head(features[-(i+2)])
                # Upsample to match main output size
                aux_out = F.interpolate(aux_out, size=main_output.shape[2:], 
                                      mode='trilinear', align_corners=False)
                aux_outputs.append(aux_out)
            return main_output, aux_outputs
        
        return main_output


class FemurSegmentationNet(nn.Module):
    """
    Custom model for femur bone segmentation from QCT DICOM images
    Combines residual learning, attention mechanisms, and multi-scale processing
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,  # background + femur
        base_channels: int = 32,
        use_deep_supervision: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Encoder
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 7, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder blocks with increasing channels
        channels = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16]
        
        self.encoder_blocks = nn.ModuleList([
            self._make_layer(channels[i], channels[i+1], 2, stride=2 if i > 0 else 1)
            for i in range(len(channels)-1)
        ])
        
        # Bridge with Dense ASPP
        self.bridge = DenseASPP(channels[-1], channels[-1])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(channels)-1, 0, -1):
            self.decoder_blocks.append(
                self._make_decoder_block(channels[i], channels[i-1])
            )
        
        # Segmentation head
        self.seg_head = BoneSegmentationHead(
            base_channels, num_classes, use_deep_supervision
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _make_decoder_block(self, in_channels: int, out_channels: int):
        return nn.ModuleDict({
            'upsample': nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ),
            'conv_block': ResidualBlock(out_channels * 2, out_channels)  # *2 for skip connection
        })
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        
        # Encoder
        skip_connections = [x]
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
        
        # Bridge
        x = self.bridge(x)
        
        # Decoder
        decoder_features = [x]
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Upsample current features
            x = decoder_block['upsample'](x)
            
            # Get corresponding skip connection
            skip = skip_connections[-(i+2)]
            
            # Ensure spatial dimensions match before concatenation
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            # Concatenate with skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Apply convolution block
            x = decoder_block['conv_block'](x)
            decoder_features.append(x)
        
        # Segmentation head
        output = self.seg_head(decoder_features)
        
        return output


def create_femur_segmentation_model(
    in_channels: int = 1,
    num_classes: int = 2,
    base_channels: int = 32,
    use_deep_supervision: bool = True,
    dropout_rate: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> FemurSegmentationNet:
    """
    Create a custom femur segmentation model optimized for QCT DICOM images
    
    Args:
        in_channels: Number of input channels (typically 1 for grayscale CT)
        num_classes: Number of output classes (2 for background + femur)
        base_channels: Base number of channels (controls model size)
        use_deep_supervision: Whether to use deep supervision for training
        dropout_rate: Dropout rate for regularization
        device: Device to place the model on
    
    Returns:
        FemurSegmentationNet model
    """
    model = FemurSegmentationNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        use_deep_supervision=use_deep_supervision,
        dropout_rate=dropout_rate
    )
    
    return model.to(device)


