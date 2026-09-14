"""
Hybrid multi-head image-quality model: a MobileNetV3-Small CNN branch fused
with a small MLP over classical image-quality features (classical_features.py),
feeding a shared trunk that produces 5 independent per-issue sigmoid heads
(blur, underexposure, overexposure, noise, corruption) plus a bounded
regression head for the overall 0-100 quality_score.

IMPORT NOTE: backend/app/ml/ is the single source of truth for this
architecture. The backend imports it directly as `app.ml.cnn_model`.
ml_training/ (a separate top-level package, not nested under backend/) is
intended to import it via a `sys.path.insert(0, ".../backend")` at the top
of each training script, so it can do `from app.ml.cnn_model import
HybridQualityModel` -- a plain relative path, not an editable pip install.
This keeps the ml stack dependency-light per BUILD_SPEC (no packaging
metadata needed for backend/ solely to satisfy ml_training/'s imports).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .classical_features import FEATURE_NAMES

# Verified against the installed torchvision (0.20.1): this is
# mobilenet_v3_small's classifier[0].in_features -- i.e. the size of
# features(x) + avgpool(x) flattened, for any input resolution. Confirmed
# both from the classifier definition and from an actual forward pass
# (features(x) -> [N, 576, 7, 7] for 224x224 input).
CNN_EMBEDDING_DIM = 576

CLASSICAL_EMBEDDING_DIM = 32
TRUNK_HIDDEN_DIM_1 = 256
TRUNK_HIDDEN_DIM_2 = 128
# Per BUILD_SPEC.md §1.5's pinned architecture: BatchNorm only after the
# first trunk layer, and a different dropout rate per layer.
TRUNK_DROPOUT_1 = 0.3
TRUNK_DROPOUT_2 = 0.2

ISSUE_HEADS = ["blur", "underexposure", "overexposure", "noise", "corruption"]
QUALITY_SCORE_HEAD = "quality_score"

# quality_score's target domain is exactly [0, 100] (see
# ml_training/data_gen/build_labels_from_kadid.py), so the regression head
# uses sigmoid*100 to bound its output to that same domain, rather than an
# unconstrained linear output that could stray outside it.
QUALITY_SCORE_MAX = 100.0


class HybridQualityModel(nn.Module):
    def __init__(self, num_classical_features: int = len(FEATURE_NAMES), pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.cnn_features = backbone.features
        self.cnn_pool = backbone.avgpool

        # Per BUILD_SPEC.md §1.5, extended from 7 to num_classical_features
        # (8, since our extract_feature_vector also includes jpeg_blockiness
        # -- see classical_features.py and the project conversation log for
        # why that deviation from §1.5's literal 7-dim list was kept).
        self.classical_branch = nn.Sequential(
            nn.Linear(num_classical_features, CLASSICAL_EMBEDDING_DIM),
            nn.BatchNorm1d(CLASSICAL_EMBEDDING_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(CLASSICAL_EMBEDDING_DIM, CLASSICAL_EMBEDDING_DIM),
            nn.ReLU(inplace=True),
        )

        trunk_input_dim = CNN_EMBEDDING_DIM + CLASSICAL_EMBEDDING_DIM
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, TRUNK_HIDDEN_DIM_1),
            nn.BatchNorm1d(TRUNK_HIDDEN_DIM_1),
            nn.ReLU(inplace=True),
            nn.Dropout(TRUNK_DROPOUT_1),
            nn.Linear(TRUNK_HIDDEN_DIM_1, TRUNK_HIDDEN_DIM_2),
            nn.ReLU(inplace=True),
            nn.Dropout(TRUNK_DROPOUT_2),
        )

        self.issue_heads = nn.ModuleDict(
            {name: nn.Linear(TRUNK_HIDDEN_DIM_2, 1) for name in ISSUE_HEADS}
        )
        self.quality_head = nn.Linear(TRUNK_HIDDEN_DIM_2, 1)

    def forward(
        self, image_tensor: torch.Tensor, classical_features_tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        image_tensor: (N, 3, H, W) float tensor, ImageNet-normalized.
        classical_features_tensor: (N, len(FEATURE_NAMES)) float tensor.
        Returns a dict with one (N,) tensor per issue in ISSUE_HEADS
        (sigmoid probabilities in [0, 1]) plus "quality_score" (in [0, 100]).
        """
        cnn_feats = self.cnn_pool(self.cnn_features(image_tensor))
        cnn_embedding = torch.flatten(cnn_feats, 1)

        classical_embedding = self.classical_branch(classical_features_tensor)

        combined = torch.cat([cnn_embedding, classical_embedding], dim=1)
        trunk_out = self.trunk(combined)

        outputs: Dict[str, torch.Tensor] = {
            name: torch.sigmoid(head(trunk_out)).squeeze(-1)
            for name, head in self.issue_heads.items()
        }
        outputs[QUALITY_SCORE_HEAD] = torch.sigmoid(self.quality_head(trunk_out)).squeeze(-1) * QUALITY_SCORE_MAX
        return outputs

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Returns the MobileNetV3-Small layer to hook for Grad-CAM: the last
        Conv2dNormActivation block in `features` (index -1), i.e. the final
        convolutional feature map before global average pooling. This is
        the standard Grad-CAM target for MobileNet-family backbones -- the
        deepest layer that still has spatial resolution to localize with.
        """
        return self.cnn_features[-1]

    @classmethod
    def from_pretrained(
        cls,
        weights_path: Union[str, Path],
        num_classical_features: int = len(FEATURE_NAMES),
    ) -> "HybridQualityModel":
        """
        Loads a trained model for inference. Instantiated with
        pretrained=False since ImageNet initialization is irrelevant once
        our own fine-tuned weights (from ml_training/export_weights.py) are
        loaded over it; sets eval() so BatchNorm/Dropout behave correctly
        at inference time.
        """
        model = cls(num_classical_features=num_classical_features, pretrained=False)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
