"""Generate the paper's fixed-checkpoint Chronos-2 forecasts from daily prefixes."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import CALIBRATION_START, read_daily, sha256

REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
CHECKSUMS = {
    "model.safetensors": "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42",
    "config.json": "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031",
}


def infer(data_path, checkpoint, output):
    import torch
    from chronos import Chronos2Pipeline

    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Choose a new output CSV and sidecar path.")
    for name, digest in CHECKSUMS.items():
        if sha256(checkpoint / name) != digest:
            raise ValueError(f"Checkpoint mismatch: {name}; use amazon/chronos-2 at {REVISION}.")
    data = read_daily(data_path)
    if len(data) <= CALIBRATION_START:
        raise ValueError("At least 15 daily rows are needed for rolling inference.")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260914)
    torch.use_deterministic_algorithms(True)
    pipeline = Chronos2Pipeline.from_pretrained(
        str(checkpoint), device_map="cpu", torch_dtype=torch.float32, local_files_only=True,
    )
    pipeline.model.eval()
    pipeline.model.requires_grad_(False)
    started = time.perf_counter()
    rows = []
    for t in range(CALIBRATION_START, len(data)):
        prefix = data.iloc[:t]
        context = pd.DataFrame({"item_id": "store", "timestamp": prefix.date,
                                "target": prefix.sales.to_numpy(np.float32)})
        with torch.inference_mode():
            prediction = pipeline.predict_df(
                context, prediction_length=1, quantile_levels=list(np.arange(1, 10) / 10),
                context_length=2048, cross_learning=False, batch_size=256, freq="D",
            ).iloc[0]
        row = {"origin_index": t, "date": str(data.date.iloc[t].date()),
               "context_end": str(prefix.date.iloc[-1].date()),
               "Chronos2": float(prediction["predictions"])}
        for q in np.arange(1, 10) / 10:
            row[f"q{int(round(q * 100))}"] = float(prediction[str(q)])
        for name in ["Chronos2"] + [f"q{i}" for i in range(10, 100, 10)]:
            if not np.isfinite(row[name]):
                raise ValueError(f"Nonfinite Chronos output at origin {t}.")
            row[name] = max(0.0, row[name])
        rows.append(row)
        if t % 21 == 0 or t == len(data) - 1:
            print(f"origin={t}, elapsed={time.perf_counter() - started:.1f}s", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    metadata = {
        "base_model": "amazon/chronos-2", "revision": REVISION,
        "checkpoint_checksums": CHECKSUMS, "data_sha256": sha256(data_path),
        "cache_sha256": sha256(output), "device": "cpu", "dtype": "float32",
        "context_length": 2048, "prediction_length": 1, "cross_learning": False,
        "rows": len(rows), "seconds": time.perf_counter() - started,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
