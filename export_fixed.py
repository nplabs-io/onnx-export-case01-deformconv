"""
export_fixed.py
===============
The fix: export the same model with ONNX opset 19, then PROVE it works.

ROOT CAUSE
  torchvision's DeformConv2d maps to the ONNX operator "DeformConv", which
  only exists from opset 19 onwards. At opset 18, torch.onnx.export still
  writes that node and reports success - producing a file that no ONNX
  runtime will load (see reproduce_error.py).

THE FIX (one argument)
  opset_version=18  ->  opset_version=19

Everything else in this file is verification. An export is not "fixed"
until the file is proven valid and numerically equivalent to PyTorch:
  [1/4] export at opset 19
  [2/4] onnx.checker (full_check) passes; DeformConv nodes are counted
  [3/4] ONNX Runtime loads and runs it on CPU (and on CUDA, if available)
  [4/4] parity: PyTorch vs ONNX, on a random input AND on a real image

WHY PARITY IS CHECKED PER UNIT
  The YOLO output [1, 84, anchors] mixes two units: rows 0-3 are box
  coordinates in PIXELS (0..640), rows 4-83 are class scores (0..1).
  A single absolute threshold is unit-blind: float32 rounding on a 637 px
  coordinate is ~1e-3 px, which would "fail" a 1e-3 limit even though it is
  1/500 of a pixel. (The unmodified yolov8n shows the same ~1.4e-3 on a real
  photo.) So each part is judged in its own unit:
        cosine similarity (whole output)  >= 0.9999
        class scores   max |difference|   <= 1e-3
        box coordinates max |difference|  <= 0.01 px
  plus: the decoded detections on bus.jpg must match one-to-one.

Outputs: outputs/after_opset19.onnx, outputs/after_report.json, outputs/after_log.txt
Exit codes: 0 = fixed and verified, 1 = a verification step failed
"""

from __future__ import annotations

import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import torch  # always import torch before onnxruntime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dcn_yolo import get_model  # noqa: E402
from yolo_post import decode, letterbox, match_detections  # noqa: E402

OPSET = 19            # <- THE FIX. DeformConv exists from opset 19 onwards.
COS_MIN = 0.9999        # direction of the whole output vector
SCORE_ABS_MAX = 1e-3    # class scores are probabilities (0..1)
BOX_ABS_MAX_PX = 1e-2   # box coordinates are pixels; 0.01 px cannot move a detection
OUT_DIR = HERE / "outputs"
LOG_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG_LINES.append(msg)


def parity(ref: np.ndarray, out: np.ndarray) -> dict:
    """PyTorch vs ONNX on a YOLO output [1, 4 + classes, anchors], judged per unit."""
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(out, dtype=np.float64)
    diff = np.abs(a - b)[0]
    cos = float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b)))
    box_max = float(diff[:4].max())     # rows 0-3: x, y, w, h in pixels
    score_max = float(diff[4:].max())   # rows 4+: class scores
    return {
        "cosine_similarity": cos,
        "box_max_abs_diff_px": box_max,
        "score_max_abs_diff": score_max,
        "unit_mixed_max_abs_diff": float(diff.max()),  # reported for transparency only
        "passed": cos >= COS_MIN and score_max <= SCORE_ABS_MAX and box_max <= BOX_ABS_MAX_PX,
    }


def run_cuda_strict(ort, path: Path, feeds: dict) -> str:
    """Runs on CUDA with CPU fallback DISABLED, so a silent fallback cannot hide a gap."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return "not available"
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    try:
        sess = ort.InferenceSession(str(path), sess_options=so,
                                    providers=["CUDAExecutionProvider"])
        sess.run(None, feeds)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"failed: {' '.join(str(exc).split())[:200]}"


def main() -> int:
    import cv2  # noqa: PLC0415
    import onnx  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415
    from ultralytics.utils import ASSETS  # noqa: PLC0415

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT_DIR / "after_opset19.onnx"
    report: dict = {
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "opset": OPSET,
        "thresholds": {"cosine_min": COS_MIN, "score_max_abs_diff": SCORE_ABS_MAX,
                       "box_max_abs_diff_px": BOX_ABS_MAX_PX},
        "note": "Numerical parity only - task-level accuracy was not measured.",
    }
    log(f"python {platform.python_version()} | torch {torch.__version__} | "
        f"onnx {onnx.__version__} | onnxruntime {ort.__version__}")
    log(f"model: YOLOv8n + DeformConv2d  |  opset {OPSET} (fix)")
    log("")

    model = get_model(HERE / "weights" / "dcn_yolov8n.pth")
    torch.manual_seed(0)
    x_random = torch.rand(1, 3, 640, 640)

    # [1/4] Export - identical to the failing call except for opset_version.
    onnx_path.unlink(missing_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (x_random,), str(onnx_path), dynamo=True,
                          opset_version=OPSET, external_data=False,
                          input_names=["images"], output_names=["output0"])
    log(f"[1/4] export        OK - {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # [2/4] Structural validity + evidence that DeformConv is really in the graph.
    onnx.checker.check_model(str(onnx_path), full_check=True)
    graph = onnx.load(str(onnx_path)).graph
    n_deform = sum(1 for node in graph.node if node.op_type == "DeformConv")
    report["deformconv_nodes"] = n_deform
    log(f"[2/4] onnx.checker  OK - valid opset-{OPSET} model, {n_deform} DeformConv nodes")

    # [3/4] ONNX Runtime: load + run on CPU, then CUDA (strict) if present.
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name  # read from the file, never assumed
    feeds = {in_name: x_random.numpy()}
    onnx_random = sess.run(None, feeds)[0]
    cuda = run_cuda_strict(ort, onnx_path, feeds)
    report["ort"] = {"cpu": "ok", "cuda": cuda}
    log(f"[3/4] onnxruntime   OK - CPU runs; CUDA: {cuda}")

    # [4/4] Parity on a random input and on a real photo (bus.jpg ships with ultralytics).
    img = cv2.imread(str(Path(ASSETS) / "bus.jpg"))
    x_img_np, ratio, pad = letterbox(img, 640)
    with torch.no_grad():
        torch_random = model(x_random).numpy()
        torch_img = model(torch.from_numpy(x_img_np)).numpy()
    onnx_img = sess.run(None, {in_name: x_img_np})[0]

    p_random = parity(torch_random, onnx_random)
    p_image = parity(torch_img, onnx_img)
    dets_torch = decode(torch_img, ratio, pad, img.shape)
    dets_onnx = decode(onnx_img, ratio, pad, img.shape)
    matched = match_detections(dets_torch, dets_onnx)
    report["parity"] = {"random_input": p_random, "bus_jpg": p_image}
    report["detections_bus_jpg"] = {"pytorch": len(dets_torch), "onnx": len(dets_onnx),
                                    "matched": matched}
    for label, p in (("random input", p_random), ("bus.jpg     ", p_image)):
        verdict = "PASSED" if p["passed"] else "FAILED"
        log(f"[4/4] parity {label} {verdict} - cosine {p['cosine_similarity']:.8f} | "
            f"boxes max |diff| {p['box_max_abs_diff_px']:.1e} px | "
            f"scores max |diff| {p['score_max_abs_diff']:.1e}")
    log(f"      detections on bus.jpg: PyTorch {len(dets_torch)}, ONNX {len(dets_onnx)}, "
        f"matched {matched}/{len(dets_torch)}")

    ok = p_random["passed"] and p_image["passed"] and matched == len(dets_torch) == len(dets_onnx)
    report["status"] = "ok" if ok else "failed"
    log("")
    log("RESULT: fixed and verified." if ok else "RESULT: verification FAILED - see above.")
    log("Note: numerical parity only - task-level accuracy was not measured.")

    (OUT_DIR / "after_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "after_log.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
