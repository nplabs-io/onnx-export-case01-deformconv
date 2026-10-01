"""
reproduce_error.py
==================
The "before" state: exports the DCN model exactly the way most people do
(torch.onnx.export, opset 18) and then tries to use the file.

Expected on torch 2.11 / onnxruntime 1.27:
  [1/3] export      -> finishes WITHOUT error (this is the trap)
  [2/3] onnx.checker -> rejects the file: DeformConv does not exist in opset 18
  [3/3] onnxruntime  -> refuses to load the model

Exit codes: 0 = failure reproduced as expected
            1 = model unexpectedly works (different environment?)
            2 = export itself crashed (a different failure than this case)
Log: outputs/before_log.txt
"""

from __future__ import annotations

import platform
import sys
import traceback
import warnings
from pathlib import Path

import torch  # always before onnxruntime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dcn_yolo import get_model  # noqa: E402

OPSET = 18
OUT_DIR = HERE / "outputs"
LOG_LINES = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG_LINES.append(msg)


def short(exc: BaseException, limit: int = 400) -> str:
    text = " ".join(str(exc).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def main() -> int:
    import onnx  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights = HERE / "weights" / "dcn_yolov8n.pth"
    onnx_path = OUT_DIR / "before_opset18.onnx"

    log(f"python {platform.python_version()} | torch {torch.__version__} | "
        f"onnx {onnx.__version__} | onnxruntime {ort.__version__}")
    log(f"model: YOLOv8n + DeformConv2d  |  opset {OPSET}")
    log("")

    model = get_model(weights)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640)

    # [1/3] export -------------------------------------------------------
    onnx_path.unlink(missing_ok=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, (x,), str(onnx_path), dynamo=True,
                              opset_version=OPSET, external_data=False)
    except Exception as exc:  # noqa: BLE001
        log(f"[1/3] export        FAILED: {type(exc).__name__}: {short(exc)}")
        log(traceback.format_exc())
        log("\nRESULT: export crashed - this is NOT the failure this case documents.")
        return 2
    size_mb = onnx_path.stat().st_size / 1e6
    log(f"[1/3] export        OK - no error raised, {onnx_path.name} written ({size_mb:.1f} MB)")

    # [2/3] checker ------------------------------------------------------
    checker_failed = False
    try:
        onnx.checker.check_model(str(onnx_path), full_check=True)
        log("[2/3] onnx.checker  OK")
    except Exception as exc:  # noqa: BLE001
        checker_failed = True
        log(f"[2/3] onnx.checker  FAILED: {short(exc)}")

    # [3/3] ONNX Runtime -------------------------------------------------
    ort_failed = False
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        sess.run(None, {sess.get_inputs()[0].name: x.numpy()})
        log("[3/3] onnxruntime   OK - model loaded and ran")
    except Exception as exc:  # noqa: BLE001
        ort_failed = True
        log(f"[3/3] onnxruntime   FAILED: {short(exc)}")

    log("")
    if checker_failed or ort_failed:
        log("RESULT: reproduced. The export reported success, but the file is not a "
            "valid opset-18 model and ONNX Runtime cannot use it.")
        code = 0
    else:
        log("RESULT: NOT reproduced - the model works in this environment.")
        code = 1

    (OUT_DIR / "before_log.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
