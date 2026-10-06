"""
network.py
Multi-domain Deep Learning Architecture for Diabetic Retinopathy:
- Spatial Encoder: EfficientNet-B0 backbone (1280-dim representation)
- Vessel Structural Encoder: 3-layer ConvNet (1 -> 32 -> 64 -> 128) + Pool + Linear(128)
- Frequency Encoder: 3-layer ConvNet (1 -> 32 -> 64 -> 128) + Pool + Linear(128)
- Feature Fusion: Concatenation (1280 / 1408 / 1536) -> MLP Classifier
- Supports Binary Screening (num_classes=2) and Five-Class Ordinal Grading (num_classes=5)
- Supports all ablation configurations (Spatial only, Spatial+Vessel, Spatial+Freq, Full)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple, Dict


class SingleDomainEncoder(nn.Module):
    """
    Lightweight 3-stage convolutional encoder for single-channel retinal maps
    (vessel structural representation or FFT frequency representation).
    Conv 1 -> 32 -> Conv 32 -> 64 -> Conv 64 -> 128 -> AdaptiveAvgPool -> Linear 128.
    """
    def __init__(self, out_features: int = 128):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Stage 1: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Stage 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Stage 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_features),
            nn.GroupNorm(num_groups=1, num_channels=out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.conv_blocks(x)
        feat_vec = self.fc(feat_map)
        return feat_vec


class DRMultiDomainModel(nn.Module):
    """
    Proposed Multi-Domain Diabetic Retinopathy Classifier.
    Fuses spatial fundus features, retinal vessel structural features, and FFT spectral features.
    """
    def __init__(
        self,
        num_classes: int = 2,
        include_vessel: bool = True,
        include_frequency: bool = True,
        dropout_rate: float = 0.3,
        pretrained_spatial: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.include_vessel = include_vessel
        self.include_frequency = include_frequency

        # 1. Spatial Backbone: EfficientNet-B0
        if pretrained_spatial:
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT
                self.spatial_backbone = models.efficientnet_b0(weights=weights)
            except Exception:
                self.spatial_backbone = models.efficientnet_b0(weights=None)
        else:
            self.spatial_backbone = models.efficientnet_b0(weights=None)

        # Spatial feature extractor: remove classification head
        self.spatial_features = self.spatial_backbone.features
        self.spatial_pool = self.spatial_backbone.avgpool
        spatial_dim = 1280

        # 2. Vessel Structural Encoder
        fusion_dim = spatial_dim
        if self.include_vessel:
            self.vessel_encoder = SingleDomainEncoder(out_features=128)
            fusion_dim += 128
        else:
            self.vessel_encoder = None

        # 3. Frequency Domain Encoder
        if self.include_frequency:
            self.frequency_encoder = SingleDomainEncoder(out_features=128)
            fusion_dim += 128
        else:
            self.frequency_encoder = None

        # 4. Multi-Domain Fusion & Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GroupNorm(num_groups=1, num_channels=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def extract_spatial_features(self, x_spatial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns spatial feature map (for Grad-CAM) and pooled feature vector."""
        feat_map = self.spatial_features(x_spatial)
        pooled = self.spatial_pool(feat_map)
        pooled_vec = torch.flatten(pooled, 1)
        return feat_map, pooled_vec

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_vessel: Optional[torch.Tensor] = None,
        x_freq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x_spatial: (B, 3, 224, 224) spatial fundus tensor.
            x_vessel: Optional (B, 1, 224, 224) vessel structural tensor.
            x_freq: Optional (B, 1, 224, 224) frequency domain tensor.
            
        Returns:
            logits: (B, num_classes).
        """
        _, spatial_vec = self.extract_spatial_features(x_spatial)
        features = [spatial_vec]

        if self.include_vessel:
            if x_vessel is None:
                raise ValueError("Model configured with include_vessel=True but x_vessel not provided.")
            vessel_vec = self.vessel_encoder(x_vessel)
            features.append(vessel_vec)

        if self.include_frequency:
            if x_freq is None:
                raise ValueError("Model configured with include_frequency=True but x_freq not provided.")
            freq_vec = self.frequency_encoder(x_freq)
            features.append(freq_vec)

        fused_vec = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        logits = self.classifier(fused_vec)
        return logits

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns the final convolutional block of the spatial backbone for Grad-CAM."""
        return self.spatial_features[-1]
