"""
build_model.py
==============
Builds weights/dcn_yolov8n.pth deterministically from the official yolov8n.pt.

What it does:
  1. Loads the official, pretrained yolov8n.pt
  2. Replaces the bottleneck 3x3 convs of layers 6 and 8 with DeformConv2d
     (the deformable conv starts from the original conv weights)
  3. Initialises the offset/mask branches so the model behaves like a
     trained DCN model: offsets have ~0.5 px RMS (non-zero, so bilinear
     sampling is really exercised) and the mask is close to 1
  4. Checks that detections on bus.jpg still match the original yolov8n
  5. Saves the checkpoint (plus a copy of the official yolov8n.pt next to it,
     which get_model() uses as the base architecture) and verifies it
     reloads bit-exactly via get_model()

HONESTY NOTE: the DCN branches are calibrated random initialisation, NOT
trained weights. The case is about export behaviour, not accuracy.

Usage:
    python build_model.py --base path/to/yolov8n.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dcn_yolo import BASE_FILE, DCN_LAYERS, IMGSZ, DCNConv, DetectOnly, apply_dcn, get_model  # noqa: E402
from yolo_post import decode, format_dets, letterbox, match_detections  # noqa: E402


@torch.no_grad()
def calibrate(det, x, offset_px: float, mask_logit_std: float, mask_bias: float):
    """Scales each DCN branch, layer by layer, on a real image."""
    stats = []
    for name, m in [(n, m) for n, m in det.named_modules() if isinstance(m, DCNConv)]:
        captured = {}
        hook = m.register_forward_pre_hook(lambda mod, inp: captured.__setitem__("x", inp[0]))
        det(x)
        hook.remove()
        f = captured["x"]

        off = F.conv2d(f, m.offset.weight, None, m.offset.stride, m.offset.padding)
        m.offset.weight.mul_(offset_px / float(off.std()))
        m.offset.bias.zero_()
        line = f"{name}: offset rms -> {offset_px} px"

        if m.mask is not None:
            logit = F.conv2d(f, m.mask.weight, None, m.mask.stride, m.mask.padding)
            m.mask.weight.mul_(mask_logit_std / float(logit.std()))
            m.mask.bias.fill_(mask_bias)
            line += f", mask ~ sigmoid({mask_bias} +/- {mask_logit_std})"
        stats.append(line)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--base", default="yolov8n.pt", help="official yolov8n.pt (downloaded if missing)")
    ap.add_argument("--out", type=Path, default=HERE / "weights" / "dcn_yolov8n.pth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mask", action="store_true", help="DCNv1 (no modulation mask)")
    ap.add_argument("--offset-px", type=float, default=0.5)
    args = ap.parse_args()

    from ultralytics import YOLO  # noqa: PLC0415
    from ultralytics.utils import ASSETS  # noqa: PLC0415

    modulated = not args.no_mask
    yolo = YOLO(args.base)
    names = {int(k): str(v) for k, v in yolo.names.items()}
    det = yolo.model.float().eval()

    img_path = Path(ASSETS) / "bus.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"ERROR: cannot read {img_path}")
        return 1
    x_np, ratio, pad = letterbox(img, IMGSZ)
    x = torch.from_numpy(x_np)

    with torch.no_grad():
        ref_dets = decode(DetectOnly(det)(x), ratio, pad, img.shape)

    torch.manual_seed(args.seed)
    replaced = apply_dcn(det, DCN_LAYERS, modulated)
    stats = calibrate(det, x, args.offset_px, mask_logit_std=0.5, mask_bias=3.0)
    det.eval()
    with torch.no_grad():
        dcn_pred = DetectOnly(det)(x)
    dcn_dets = decode(dcn_pred, ratio, pad, img.shape)

    print(f"Replaced {len(replaced)} conv layers with DeformConv2d "
          f"({'DCNv2, modulated' if modulated else 'DCNv1'}):")
    for s in stats:
        print(f"  {s}")
    print(f"\nbus.jpg - original yolov8n ({len(ref_dets)} detections):")
    for line in format_dets(ref_dets, names):
        print(f"  {line}")
    print(f"bus.jpg - DCN model ({len(dcn_dets)} detections):")
    for line in format_dets(dcn_dets, names):
        print(f"  {line}")
    matched = match_detections(ref_dets, dcn_dets)
    print(f"Matched (same class, IoU > 0.5): {matched}/{len(ref_dets)}")

    meta = {
        "base_file": BASE_FILE, "nc": len(names), "names": names,
        "dcn_layers": list(DCN_LAYERS), "modulated": modulated, "imgsz": IMGSZ,
        "replaced": replaced, "seed": args.seed, "offset_px": args.offset_px,
        "note": "DCN branches are calibrated random initialisation, not trained weights.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base_src = Path(getattr(yolo, "ckpt_path", None) or args.base).resolve()
    base_dst = (args.out.parent / BASE_FILE).resolve()
    if base_src != base_dst:
        shutil.copy2(base_src, base_dst)
    torch.save({"state_dict": det.state_dict(), "meta": meta}, args.out)

    reloaded = get_model(args.out)
    with torch.no_grad():
        diff = float((reloaded(x) - dcn_pred).abs().max())
    print(f"\nSaved: {args.out}  (base: {base_dst.name})")
    print(f"Round-trip via get_model(): max abs diff = {diff:.3e}")
    if diff > 1e-5:
        print("ERROR: reloaded model differs from the built model")
        return 1
    print("BUILD OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
