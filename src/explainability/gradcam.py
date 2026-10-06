"""
gradcam.py
Grad-CAM (Gradient-weighted Class Activation Mapping) for the spatial EfficientNet-B0 backbone.
Extracts spatial activation maps, weights them by output gradients, and generates
superimposed visual explanations.

Note: Grad-CAM is an explainability / saliency visualization tool. It does NOT constitute
clinical diagnostic proof, automated lesion detection, or independently validated pathology localization.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
from src.models.network import DRMultiDomainModel


class GradCAM:
    """
    Grad-CAM implementation targeting the final convolutional stage of EfficientNet-B0.
    """
    def __init__(self, model: DRMultiDomainModel, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or model.get_gradcam_target_layer()
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple where the first element is the gradient w.r.t layer output
            self.gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(
        self,
        x_spatial: torch.Tensor,
        x_vessel: Optional[torch.Tensor] = None,
        x_freq: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates Grad-CAM heatmap for the specified or predicted target class.
        
        Args:
            x_spatial: (1, 3, 224, 224) torch tensor
            x_vessel: Optional (1, 1, 224, 224) torch tensor
            x_freq: Optional (1, 1, 224, 224) torch tensor
            target_class: Target class index. If None, uses model argmax prediction.
            
        Returns:
            heatmap: (224, 224) float32 in range [0, 1]
            predicted_class: class index
            confidence: softmax probability for predicted class
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(x_spatial, x_vessel, x_freq)
        probs = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        confidence = float(probs[0, target_class].item())

        # Backward pass
        score.backward(retain_graph=True)

        # Global average pooling of gradients: weights alpha_k
        # gradients shape: (1, C, H, W)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weighted combination of activation maps
        # activations shape: (1, C, H, W)
        activations = self.activations[0]
        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        # Heatmap = ReLU(sum_k alpha_k * A^k)
        heatmap = torch.sum(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0.0)

        # Normalize to [0, 1]
        max_val = np.max(heatmap)
        if max_val > 1e-7:
            heatmap = heatmap / max_val
        else:
            heatmap = np.zeros_like(heatmap)

        # Resize heatmap to match spatial input dimensions (224, 224)
        heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

        return heatmap_resized, target_class, confidence

    def generate_overlay(
        self,
        original_image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Creates a color-mapped visualization overlay on the original fundus image.
        
        Args:
            original_image_rgb: (224, 224, 3) uint8 or float in [0, 1] or [0, 255]
            heatmap: (224, 224) float32 in [0, 1]
            alpha: Blend factor for overlay
            colormap: OpenCV colormap (default: COLORMAP_JET)
            
        Returns:
            overlay_rgb: (224, 224, 3) uint8 RGB image
        """
        if original_image_rgb.max() <= 1.0:
            base_img = (original_image_rgb * 255.0).astype(np.uint8)
        else:
            base_img = original_image_rgb.astype(np.uint8)

        # Convert heatmap to uint8 colormap
        heatmap_uint8 = (heatmap * 255.0).astype(np.uint8)
        heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend
        overlay = cv2.addWeighted(base_img, 1.0 - alpha, heatmap_color_rgb, alpha, 0)
        return overlay

    def remove_hooks(self):
        """Removes registered PyTorch hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
