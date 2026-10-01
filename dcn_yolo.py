"""
dcn_yolo.py
===========
YOLOv8n with deformable convolution (DCNv2) in the deep backbone stages.

This reproduces a very common community modification ("YOLOv8 + DCN"):
the 3x3 conv inside every C2f bottleneck of backbone layers 6 (P4, 40x40)
and 8 (P5, 20x20) is replaced by torchvision's DeformConv2d. Each replaced
layer gets a small offset branch (where to sample) and, for DCNv2, a mask
branch (how much to trust each sample).

get_model(weights) is the single loader entry point. It needs no internet:
the base architecture is loaded from the official yolov8n.pt that
build_model.py copies next to the checkpoint, then the DCN layers are
applied and the checkpoint weights loaded.

Why not rebuild from yolov8n.yaml? On ultralytics 8.4 a yaml-built yolov8n
does NOT behave like the pretrained yolov8n.pt even with identical weights:
the outputs diverge from the SPPF block (layer 9) onwards, because a
non-weight setting of that block changed in newer ultralytics versions.
Loading the architecture from the official .pt avoids that trap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

BASE_FILE = "yolov8n.pt"  # official weights, copied next to the checkpoint
DCN_LAYERS = (6, 8)  # yolov8n: C2f blocks at P4 and P5
IMGSZ = 640


class DCNConv(nn.Module):
    """
    Drop-in replacement for an ultralytics Conv (conv -> bn -> act).
    The deformable conv starts from the ORIGINAL conv weights, so the
    pretrained features are kept; bn and act are reused as-is.
    """

    def __init__(self, src: nn.Module, modulated: bool = True):
        super().__init__()
        conv: nn.Conv2d = src.conv
        k = conv.kernel_size
        n_points = k[0] * k[1]
        self.offset = nn.Conv2d(conv.in_channels, 2 * n_points, k, conv.stride,
                                conv.padding, bias=True)
        self.mask = (nn.Conv2d(conv.in_channels, n_points, k, conv.stride,
                               conv.padding, bias=True) if modulated else None)
        self.dcn = DeformConv2d(conv.in_channels, conv.out_channels, k,
                                stride=conv.stride, padding=conv.padding,
                                dilation=conv.dilation, groups=conv.groups,
                                bias=conv.bias is not None)
        with torch.no_grad():
            self.dcn.weight.copy_(conv.weight)
            if conv.bias is not None:
                self.dcn.bias.copy_(conv.bias)
        self.bn = getattr(src, "bn", nn.Identity())
        self.act = getattr(src, "act", nn.Identity())

    def forward(self, x):
        offset = self.offset(x)
        mask = torch.sigmoid(self.mask(x)) if self.mask is not None else None
        return self.act(self.bn(self.dcn(x, offset, mask)))


def apply_dcn(det_model: nn.Module, layers: Sequence[int] = DCN_LAYERS,
              modulated: bool = True) -> List[str]:
    """Swaps the bottleneck 3x3 convs of the given C2f layers. Returns replaced names."""
    from ultralytics.nn.modules import C2f, Conv  # noqa: PLC0415

    replaced = []
    for i in layers:
        block = det_model.model[i]
        if not isinstance(block, C2f):
            raise TypeError(f"layer {i} is {type(block).__name__}, expected C2f")
        for j, bottleneck in enumerate(block.m):
            if not isinstance(bottleneck.cv2, Conv):
                raise TypeError(f"layer {i}.m.{j}.cv2 is not an ultralytics Conv")
            bottleneck.cv2 = DCNConv(bottleneck.cv2, modulated)
            replaced.append(f"model.{i}.m.{j}.cv2")
    return replaced


class DetectOnly(nn.Module):
    """
    Returns only the decoded prediction tensor [1, 4 + num_classes, anchors].
    (In eval mode the ultralytics head returns a tuple; exporters and parity
    tests need a single, well-defined output.)
    """

    def __init__(self, det_model: nn.Module, imgsz: int = IMGSZ,
                 names: Optional[Dict[int, str]] = None):
        super().__init__()
        self.model = det_model
        self.imgsz = imgsz
        self.names = names or {}

    def forward(self, x):
        y = self.model(x)
        return y[0] if isinstance(y, (list, tuple)) else y


def get_model(weights=None) -> nn.Module:
    """Loader entry point: official yolov8n.pt + DCN layers + checkpoint weights."""
    from ultralytics import YOLO  # noqa: PLC0415

    if weights is None:
        raise ValueError("weights path is required (run build_model.py first)")
    weights = Path(weights)
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    base = weights.parent / meta["base_file"]
    if not base.is_file():
        raise FileNotFoundError(f"{base} is missing - run build_model.py first")
    det = YOLO(str(base)).model.float().eval()
    apply_dcn(det, tuple(meta["dcn_layers"]), meta["modulated"])
    det.load_state_dict(ckpt["state_dict"], strict=True)
    det.eval()
    return DetectOnly(det, meta["imgsz"], meta["names"]).eval()
