import torch
import torch.nn as nn

from src.models.projection import (
    AcousticProjection,
    WavLMProjection
)

from src.models.fusion import (
    FeatureFusion
)

from src.models.transformer import (
    TemporalTransformer
)

from src.models.attention import (
    AttentionPooling
)

from src.models.global_fusion import (
    GlobalFeatureFusion
)

from src.models.classifier import (
    FatigueClassifier
)


class FatigueModel(nn.Module):

    def __init__(
        self,
        num_classes=3
    ):
        super().__init__()

        # ====================================================
        # FEATURE PROJECTIONS
        # ====================================================

        self.acoustic_projection = (
            AcousticProjection(
                input_dim=40,
                output_dim=128
            )
        )

        self.wavlm_projection = (
            WavLMProjection(
                input_dim=768,
                output_dim=128
            )
        )

        # ====================================================
        # TEMPORAL FEATURE FUSION
        # ====================================================

        self.feature_fusion = (
            FeatureFusion(
                acoustic_dim=128,
                wavlm_dim=128
            )
        )

        # ====================================================
        # TRANSFORMER
        # ====================================================

        self.transformer = (
            TemporalTransformer(
                input_dim=256,
                num_heads=4,
                num_layers=2,
                feedforward_dim=512,
                dropout=0.1
            )
        )

        # ====================================================
        # ATTENTION POOLING
        # ====================================================

        self.attention_pooling = (
            AttentionPooling(
                input_dim=256
            )
        )

        # ====================================================
        # GLOBAL FEATURE FUSION
        # ====================================================

        self.global_fusion = (
            GlobalFeatureFusion(
                temporal_dim=256,
                global_dim=12,
                output_dim=268
            )
        )

        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = (
            FatigueClassifier(
                input_dim=268,
                hidden_dim1=128,
                hidden_dim2=64,
                num_classes=num_classes,
                dropout=0.3
            )
        )

    def forward(
        self,
        acoustic,
        wavlm,
        global_features,
        attention_mask=None
    ):

        # ====================================================
        # ACOUSTIC PROJECTION
        # ====================================================

        acoustic = (
            self.acoustic_projection(
                acoustic
            )
        )

        # Shape:
        # [B,T,40]
        #      ↓
        # [B,T,128]

        # ====================================================
        # WAVLM PROJECTION
        # ====================================================

        wavlm = (
            self.wavlm_projection(
                wavlm
            )
        )

        # Shape:
        # [B,T,768]
        #      ↓
        # [B,T,128]

        # ====================================================
        # FEATURE FUSION
        # ====================================================

        fused = self.feature_fusion(
            acoustic,
            wavlm
        )

        # [B,T,256]

        # ====================================================
        # TRANSFORMER
        # ====================================================

        # PyTorch Transformer expects:
        #
        # src_key_padding_mask:
        # True  = padding
        # False = valid
        #
        # Our DataLoader uses:
        #
        # True  = valid
        # False = padding
        #
        # Therefore invert it.

        padding_mask = None

        if attention_mask is not None:

            padding_mask = ~attention_mask

        transformed = self.transformer(
            fused,
            src_key_padding_mask=padding_mask
        )

        # [B,T,256]

        # ====================================================
        # ATTENTION POOLING
        # ====================================================

        pooled, attention_weights = (
            self.attention_pooling(
                transformed
            )
        )

        # pooled:
        # [B,256]

        # ====================================================
        # GLOBAL FEATURE FUSION
        # ====================================================

        fused_global = self.global_fusion(
            pooled,
            global_features
        )

        # [B,267]

        # ====================================================
        # CLASSIFIER
        # ====================================================

        logits = self.classifier(
            fused_global
        )

        # [B,3]

        return {
            "logits": logits,
            "temporal_representation": pooled,
            "attention_weights": attention_weights
        }