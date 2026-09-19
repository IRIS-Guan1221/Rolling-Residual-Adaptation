"""Command-line installation demo, Chronos inference, and rolling evaluation."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform


def run(data_path, cache_path, output, ablations=False, score_start=42, window_size=21):
    from .data import load_inputs, sha256
    from .evaluate import configurations, replay, score
    from .model import RRAForecaster

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    data, cache, provenance = load_inputs(data_path, cache_path)
    configs = configurations(ablations)
    predictions = replay(data, cache, configs)
    scores = score(predictions, ["Chronos2"] + list(configs), score_start, window_size)
    if provenance.get("synthetic"):
        predictions = predictions.rename(columns={"Chronos2": "Synthetic_base"})
        scores["model"] = scores["model"].replace({"Chronos2": "Synthetic_base"})
    metadata = {
        "configuration": {name: RRAForecaster(c).configuration() for name, c in configs.items()},
        "base_provenance": provenance,
        "data_sha256": sha256(data_path), "cache_sha256": sha256(cache_path),
        "calibration_start_index": 14, "score_start_index": score_start,
        "window_size": window_size, "python": platform.python_version(),
        "versions": {n: importlib.metadata.version(n) for n in ["rra-sales", "numpy", "pandas", "lightgbm", "scikit-learn"]},
    }
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    scores.to_csv(output / "metrics.csv", index=False)
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(scores.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate", help="Replay cached base forecasts and train the RRA adapter.")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--cache", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--ablations", action="store_true")
    evaluate.add_argument("--score-start", type=int, default=42)
    evaluate.add_argument("--window-size", type=int, default=21)
    infer = sub.add_parser("infer", help="Compute prefix-only Chronos-2 forecasts and their provenance.")
    infer.add_argument("--data", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    demo = sub.add_parser("demo", help="Run a synthetic example without downloading model weights.")
    demo.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        run(args.data, args.cache, args.output, args.ablations, args.score_start, args.window_size)
    elif args.command == "infer":
        from .chronos import infer as infer_chronos
        infer_chronos(args.data, args.checkpoint, args.output)
    else:
        from .demo import write_demo
        data, cache = write_demo(args.output)
        print("Synthetic installation example: the base predictor is a trailing mean, not Chronos-2.")
        run(data, cache, args.output / "results", ablations=True)
