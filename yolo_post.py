"""
yolo_post.py
============
Minimal YOLOv8 pre/post-processing used by the case scripts:
letterbox -> model -> decode (confidence filter + class-wise NMS) -> original
image coordinates. Works on both PyTorch tensors and ONNX Runtime outputs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision


def letterbox(img_bgr: np.ndarray, size: int = 640, pad_value: int = 114):
    """Resize keeping aspect ratio, pad to size x size. Returns (x, ratio, (pad_x, pad_y))."""
    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, NCHW
    return np.ascontiguousarray(x), r, (left, top)


def decode(pred, ratio: float, pad: Tuple[int, int], orig_shape,
           conf: float = 0.25, iou: float = 0.45) -> List[Dict]:
    """[1, 4+nc, N] prediction -> list of {cls, conf, box[x1,y1,x2,y2]} in original pixels."""
    p = pred if torch.is_tensor(pred) else torch.from_numpy(np.asarray(pred))
    p = p.detach().float().cpu()[0].transpose(0, 1)  # (N, 4+nc)
    scores, cls = p[:, 4:].max(1)
    keep = scores > conf
    p, scores, cls = p[keep], scores[keep], cls[keep]
    xy, wh = p[:, :2], p[:, 2:4]
    boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)
    idx = torchvision.ops.batched_nms(boxes, scores, cls, iou)
    boxes, scores, cls = boxes[idx].clone(), scores[idx], cls[idx]

    h, w = orig_shape[:2]
    boxes[:, 0] = ((boxes[:, 0] - pad[0]) / ratio).clamp(0, w)
    boxes[:, 2] = ((boxes[:, 2] - pad[0]) / ratio).clamp(0, w)
    boxes[:, 1] = ((boxes[:, 1] - pad[1]) / ratio).clamp(0, h)
    boxes[:, 3] = ((boxes[:, 3] - pad[1]) / ratio).clamp(0, h)
    return [{"cls": int(c), "conf": float(s), "box": [float(v) for v in b]}
            for b, s, c in zip(boxes, scores, cls)]


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_detections(ref: List[Dict], other: List[Dict], iou_thr: float = 0.5) -> int:
    """How many reference detections have a same-class partner with IoU > iou_thr."""
    return sum(
        any(o["cls"] == r["cls"] and box_iou(r["box"], o["box"]) > iou_thr for o in other)
        for r in ref
    )


def format_dets(dets: List[Dict], names: Dict[int, str]) -> List[str]:
    return [f"{names.get(d['cls'], d['cls'])!s:<12} {d['conf']:.2f}  "
            f"[{', '.join(f'{v:.0f}' for v in d['box'])}]" for d in dets]
