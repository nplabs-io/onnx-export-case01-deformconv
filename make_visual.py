"""
make_visual.py
==============
Builds the case's main before/after image (1280x769, the Fiverr gig ratio).

Every number and status on the image is READ from evidence files - nothing
is typed in by hand:
  outputs/before_log.txt       <- reproduce_error.py   (the failure)
  outputs/after_report.json    <- export_fixed.py      (the verified fix)
  outputs/after_opset19.onnx   <- export_fixed.py      (detections drawn by ONNX Runtime)
  kit_runs/after/result.json   <- EdgeForge bench      (optional: latency numbers)

If the evidence is missing or does not show what the image claims, the
script stops instead of drawing a misleading picture.

Output: outputs/case01_before_after.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch  # before onnxruntime
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from yolo_post import decode, letterbox  # noqa: E402

W, H = 1280, 769
BG, PANEL, TERM = (15, 23, 42), (30, 41, 59), (2, 6, 23)
WHITE, GREY, DIM = (241, 245, 249), (148, 163, 184), (100, 116, 139)
RED, GREEN, AMBER = (248, 113, 113), (74, 222, 128), (251, 191, 36)
BOX_COLORS = [(45, 212, 191), (251, 191, 36), (167, 139, 250), (244, 114, 182)]


def font(size: int, kind: str = "sans"):
    """DejaVu fonts ship with matplotlib (an ultralytics dependency) -> same look on every OS."""
    import matplotlib  # noqa: PLC0415

    name = {"sans": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf",
            "mono": "DejaVuSansMono.ttf", "monobold": "DejaVuSansMono-Bold.ttf"}[kind]
    path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", name)
    return ImageFont.truetype(path, size)


def need(cond: bool, msg: str) -> None:
    if not cond:
        print(f"ERROR: {msg}")
        raise SystemExit(1)


def load_evidence(kit_result: Path) -> dict:
    before = HERE / "outputs" / "before_log.txt"
    after = HERE / "outputs" / "after_report.json"
    need(before.is_file(), "outputs/before_log.txt missing - run reproduce_error.py first")
    need(after.is_file(), "outputs/after_report.json missing - run export_fixed.py first")

    log = before.read_text(encoding="utf-8")
    need("[1/3] export        OK" in log, "before_log.txt does not show a 'successful' export")
    need("[2/3] onnx.checker  FAILED" in log and "DeformConv" in log,
         "before_log.txt does not show the DeformConv checker failure")
    need("[3/3] onnxruntime   FAILED" in log, "before_log.txt does not show the runtime failure")

    rep = json.loads(after.read_text(encoding="utf-8"))
    need(rep.get("status") == "ok", "after_report.json status is not ok - the fix is not verified")

    latency = {}
    if kit_result.is_file():
        measured = json.loads(kit_result.read_text(encoding="utf-8"))["benchmark"]["measured"]
        for alias in ("cuda", "cpu"):
            entry = measured.get(alias)
            if entry and entry.get("status") == "ok":
                latency[alias] = entry["latency_ms"]["p50"]
    return {"after": rep, "latency": latency}


def draw_detections(names: dict) -> Image.Image:
    """Runs the FIXED ONNX model on bus.jpg and draws its detections."""
    import cv2  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415
    from ultralytics.utils import ASSETS  # noqa: PLC0415

    img = cv2.imread(str(Path(ASSETS) / "bus.jpg"))
    x, ratio, pad = letterbox(img, 640)
    sess = ort.InferenceSession(str(HERE / "outputs" / "after_opset19.onnx"),
                                providers=["CPUExecutionProvider"])
    pred = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    dets = decode(pred, ratio, pad, img.shape)

    pil = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    f = font(36, "bold")
    placed = []  # label rectangles already drawn, to avoid overlaps
    for det in sorted(dets, key=lambda k: -k["conf"]):
        color = BOX_COLORS[det["cls"] % len(BOX_COLORS)]
        x1, y1, x2, y2 = det["box"]
        d.rectangle([x1, y1, x2, y2], outline=color, width=8)
        label = f" {names.get(det['cls'], det['cls'])} {det['conf']:.2f} "
        tw = d.textlength(label, font=f)
        lx = max(0, min(x1, pil.width - tw))   # keep the label inside the photo
        ty = max(0, y1 - 46)                    # preferred: just above the box
        while any(lx < r[2] and lx + tw > r[0] and ty < r[3] and ty + 46 > r[1] for r in placed):
            ty += 48                            # overlap -> slide down into the box
        placed.append((lx, ty, lx + tw, ty + 46))
        d.rectangle([lx, ty, lx + tw, ty + 46], fill=color)
        d.text((lx, ty + 3), label, font=f, fill=(15, 23, 42))
    return pil


def badge(d, xy, text, color):
    f = font(17, "bold")
    x, y = xy
    tw = d.textlength(text, font=f)
    d.rounded_rectangle([x, y, x + tw + 24, y + 30], radius=8, fill=color)
    d.text((x + 12, y + 5), text, font=f, fill=BG)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the before/after image.")
    ap.add_argument("--kit-result", type=Path, default=HERE / "kit_runs" / "after" / "result.json",
                    help="EdgeForge result.json with benchmark numbers (optional)")
    args = ap.parse_args()

    ev = load_evidence(args.kit_result)
    rep, latency = ev["after"], ev["latency"]
    meta = torch.load(str(HERE / "weights" / "dcn_yolov8n.pth"), map_location="cpu",
                      weights_only=True)["meta"]

    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)

    # --- title ----------------------------------------------------------
    d.text((48, 30), "PyTorch  \u2192  ONNX export rescue", font=font(20, "bold"), fill=AMBER)
    d.text((48, 58), "The export said OK. The model was broken.", font=font(40, "bold"), fill=WHITE)
    d.text((48, 110), "YOLOv8n + deformable convolution (DCNv2)  \u00b7  diagnosed, fixed and verified",
           font=font(19), fill=GREY)

    top, bottom = 160, 700
    # --- BEFORE panel ---------------------------------------------------
    lx1, lx2 = 48, 548
    d.rounded_rectangle([lx1, top, lx2, bottom], radius=16, fill=PANEL)
    badge(d, (lx1 + 24, top + 22), "BEFORE  \u00b7  opset 18", RED)
    d.rounded_rectangle([lx1 + 24, top + 72, lx2 - 24, top + 360], radius=10, fill=TERM)
    mono, monob = font(16, "mono"), font(16, "monobold")
    rows = [
        ("$ torch.onnx.export(..., opset_version=18)", GREY, mono),
        ("", GREY, mono),
        ("[1/3] export        OK", AMBER, monob),
        ("      no error raised - looks done", DIM, mono),
        ("[2/3] onnx.checker  FAILED", RED, monob),
        ("      No Op registered for DeformConv", GREY, mono),
        ("      with domain_version of 18", GREY, mono),
        ("[3/3] onnxruntime   FAILED", RED, monob),
        ("      INVALID_GRAPH: invalid model", GREY, mono),
    ]
    y = top + 90
    for text, color, f in rows:
        d.text((lx1 + 44, y), text, font=f, fill=color)
        y += 29
    d.text((lx1 + 24, top + 385), "A silent failure:", font=font(22, "bold"), fill=WHITE)
    for i, line in enumerate(["the file is written, but it is not valid", "ONNX and ONNX Runtime refuses it.",
                              "DeformConv only exists from opset 19."]):
        d.text((lx1 + 24, top + 420 + i * 30), line, font=font(20), fill=GREY)

    # --- AFTER panel ----------------------------------------------------
    rx1, rx2 = 580, 1232
    d.rounded_rectangle([rx1, top, rx2, bottom], radius=16, fill=PANEL)
    badge(d, (rx1 + 24, top + 22), "AFTER  \u00b7  opset 19", GREEN)

    det_img = draw_detections(meta["names"])
    img_h = bottom - top - 96
    img_w = int(det_img.width * img_h / det_img.height)
    canvas.paste(det_img.resize((img_w, img_h), Image.LANCZOS), (rx1 + 24, top + 72))

    sx = rx1 + 24 + img_w + 28
    pb = rep["parity"]["bus_jpg"]
    det = rep["detections_bus_jpg"]
    runtimes = "CPU + CUDA" if rep["ort"]["cuda"] == "ok" else "CPU"
    stats = [
        ("Valid ONNX", "PASSED", GREEN),
        ("ONNX Runtime", runtimes, GREEN),
        ("Parity vs PyTorch", "PASSED", GREEN),
        ("  boxes, max diff", f"{pb['box_max_abs_diff_px']:.3f} px", WHITE),
        ("  scores, max diff", f"{pb['score_max_abs_diff']:.1e}", WHITE),
        ("Same detections", f"{det['matched']}/{det['pytorch']}", GREEN),
    ]
    if "cuda" in latency:
        stats.append(("GPU latency (p50)", f"{latency['cuda']:.1f} ms", WHITE))
    if "cpu" in latency:
        stats.append(("CPU latency (p50)", f"{latency['cpu']:.1f} ms", WHITE))
    y = top + 80
    for label, value, color in stats:
        d.text((sx, y), label, font=font(17), fill=GREY)
        d.text((sx, y + 22), value, font=font(22, "bold"), fill=color)
        y += 56

    # --- honesty footer (Charter 4.1) -----------------------------------
    foot = "Numerical parity verified; task-level accuracy not measured."
    if latency:
        foot += "  Latency measured on the test machine (x86 CPU + RTX 4070 Laptop), not on target devices."
    d.text((48, 722), foot, font=font(15), fill=DIM)

    out = HERE / "outputs" / "case01_before_after.png"
    canvas.save(out)
    print(f"Saved: {out}  ({W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
