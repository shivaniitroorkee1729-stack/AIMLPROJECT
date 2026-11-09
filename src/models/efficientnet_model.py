"""
EfficientNet-B0 Model Architecture for 9-Class Navarasa Classification.
Supports:
- Pretrained ImageNet weights
- Custom classification head with Dropout regularization
- Two-stage training (Phase 1: Head training; Phase 2: Backbone fine-tuning)
- Feature map and gradient extraction hooks for Grad-CAM
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import NAVARASA_CLASSES, DEVICE, EFFICIENTNET_BEST_MODEL_PATH


class NavarasaEfficientNet(nn.Module):
    """
    Fine-tuned EfficientNet-B0 for Navarasa expression classification.
    """

    def __init__(self, num_classes=len(NAVARASA_CLASSES), dropout_p=0.4, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        # Load backbone
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = efficientnet_b0(weights=weights)

        # Feature extractor
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # Classifier head
        in_features = base_model.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_p, inplace=True),
            nn.Linear(in_features, num_classes)
        )

        # Hooks storage for Grad-CAM
        self.gradients = None
        self.activations = None
        self._register_gradcam_hooks()

    def _register_gradcam_hooks(self):
        """Registers hooks on the final convolutional layer of the feature backbone."""
        target_layer = self.features[-1]

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def freeze_backbone(self):
        """Freezes all layers in the feature backbone (Phase 1 training)."""
        for param in self.features.parameters():
            param.requires_grad = False
        print("Backbone frozen. Only classification head is trainable.")

    def unfreeze_top_layers(self, num_blocks=2):
        """
        Unfreezes the top N blocks of the backbone for fine-tuning (Phase 2 training).
        EfficientNet-B0 has 9 blocks (0 to 8).
        """
        # First freeze all
        for param in self.features.parameters():
            param.requires_grad = False

        # Unfreeze top blocks (e.g. features[7] and features[8])
        for idx in range(len(self.features) - num_blocks, len(self.features)):
            for param in self.features[idx].parameters():
                param.requires_grad = True

        # Classifier head is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Unfrozen top {num_blocks} blocks. Trainable params: {trainable_params:,} / {total_params:,}")

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def predict_probabilities(self, x):
        """Returns softmax probabilities for batch tensor x."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


def load_efficientnet_model(checkpoint_path=EFFICIENTNET_BEST_MODEL_PATH, device=DEVICE):
    """Loads trained EfficientNet model checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    model = NavarasaEfficientNet(pretrained=False)
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded EfficientNet-B0 checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, using ImageNet pretrained weights.")
        model = NavarasaEfficientNet(pretrained=True)
    model = model.to(device)
    model.eval()
    return model
