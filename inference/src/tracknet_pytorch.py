"""
PyTorch implementation of TrackNetV4 (TypeA / TypeB) for ball tracking inference.
Matches the state-dict produced by the original training code so that
.pth checkpoint files can be loaded directly with model.load_state_dict().
"""

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
    )


class MotionPromptLayer(nn.Module):
    """Generates per-frame attention maps from frame differences (power-normalised)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _power_norm(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            (5.0 / (0.45 * torch.abs(torch.tanh(a)) + 1e-1))
            * (torch.abs(x) - 0.6 * torch.tanh(b))
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imgs: (B, 9, H, W) – three concatenated RGB frames.
        Returns:
            attention: (B, 2, H, W) – two inter-frame attention maps.
        """
        # Reshape to (B, 3_frames, 3_channels, H, W)
        B, _, H, W = imgs.shape
        frames = imgs.view(B, 3, 3, H, W)

        # Normalise back to ~[0,1] (mirrors TF code: x * 0.225 + 0.45)
        norm = frames * 0.225 + 0.45

        # Convert to grayscale: weighted sum over channel dim
        gray_weights = torch.tensor(
            [0.299, 0.587, 0.114], device=imgs.device, dtype=imgs.dtype
        )
        gray = torch.einsum("bfchw,c->bfhw", norm, gray_weights)  # (B,3,H,W)

        # Frame differences → 2 maps
        diff = gray[:, 1:] - gray[:, :-1]  # (B,2,H,W)
        attention = self._power_norm(diff, self.a, self.b)
        return attention


class TrackNetV4(nn.Module):
    """UNet-style ball detector with motion-prompt fusion."""

    def __init__(
        self,
        input_height: int = 288,
        input_width: int = 512,
        fusion_type: str = "TypeA",
    ) -> None:
        super().__init__()
        self.fusion_type = fusion_type
        self.motion_prompt = MotionPromptLayer()

        # ── Encoder ────────────────────────────────────────────
        self.conv1  = _conv_bn_relu(9,   64)
        self.conv2  = _conv_bn_relu(64,  64)   # → skip1
        self.pool1  = nn.MaxPool2d(2, 2)

        self.conv3  = _conv_bn_relu(64,  128)
        self.conv4  = _conv_bn_relu(128, 128)  # → skip2
        self.pool2  = nn.MaxPool2d(2, 2)

        self.conv5  = _conv_bn_relu(128, 256)
        self.conv6  = _conv_bn_relu(256, 256)
        self.conv7  = _conv_bn_relu(256, 256)  # → skip3
        self.pool3  = nn.MaxPool2d(2, 2)

        self.conv8  = _conv_bn_relu(256, 512)
        self.conv9  = _conv_bn_relu(512, 512)
        self.conv10 = _conv_bn_relu(512, 512)

        # ── Decoder ────────────────────────────────────────────
        self.up1    = nn.Upsample(scale_factor=2)
        self.conv11 = _conv_bn_relu(768, 256)  # 512+256
        self.conv12 = _conv_bn_relu(256, 256)
        self.conv13 = _conv_bn_relu(256, 256)

        self.up2    = nn.Upsample(scale_factor=2)
        self.conv14 = _conv_bn_relu(384, 128)  # 256+128
        self.conv15 = _conv_bn_relu(128, 128)

        self.up3    = nn.Upsample(scale_factor=2)
        self.conv16 = _conv_bn_relu(192, 64)   # 128+64
        self.conv17 = _conv_bn_relu(64,  64)

        self.conv18 = nn.Sequential(nn.Conv2d(64, 3, 1))

    # ── Fusion helpers ─────────────────────────────────────────
    def _fuse_type_a(
        self, feat: torch.Tensor, attn: torch.Tensor
    ) -> torch.Tensor:
        """feat: (B,3,H,W), attn: (B,2,H,W)"""
        out = torch.stack(
            [feat[:, 0], feat[:, 1] * attn[:, 0], feat[:, 2] * attn[:, 1]],
            dim=1,
        )
        return out

    def _fuse_type_b(
        self, feat: torch.Tensor, attn: torch.Tensor
    ) -> torch.Tensor:
        out = torch.stack(
            [
                feat[:, 0] * attn[:, 0],
                feat[:, 1] * ((attn[:, 0] + attn[:, 1]) / 2),
                feat[:, 2] * attn[:, 1],
            ],
            dim=1,
        )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 9, H, W) – three concatenated RGB frames (float, 0-1).
        Returns:
            out: (B, 3, H, W) – per-frame heatmaps after sigmoid.
        """
        attn = self.motion_prompt(x)       # (B, 2, H, W)

        # Encoder
        x1 = self.conv2(self.conv1(x))     # skip1
        x2 = self.conv4(self.conv3(self.pool1(x1)))  # skip2
        x3 = self.conv7(self.conv6(self.conv5(self.pool2(x2))))  # skip3
        x  = self.conv10(self.conv9(self.conv8(self.pool3(x3))))

        # Decoder
        x  = self.conv13(self.conv12(self.conv11(
            torch.cat([self.up1(x), x3], dim=1)
        )))
        x  = self.conv15(self.conv14(
            torch.cat([self.up2(x), x2], dim=1)
        ))
        x  = self.conv17(self.conv16(
            torch.cat([self.up3(x), x1], dim=1)
        ))

        x  = self.conv18(x)               # (B, 3, H, W)

        # Fusion
        if self.fusion_type == "TypeA":
            x = self._fuse_type_a(x, attn)
        else:
            x = self._fuse_type_b(x, attn)

        return torch.sigmoid(x)


def get_model(
    model_name: str,
    input_height: int = 288,
    input_width: int = 512,
) -> TrackNetV4:
    """Factory matching the signature used by ball_tracker."""
    if "TypeB" in model_name:
        fusion = "TypeB"
    else:
        fusion = "TypeA"
    return TrackNetV4(input_height, input_width, fusion_type=fusion)
