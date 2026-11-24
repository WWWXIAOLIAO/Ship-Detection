"""Validation selection is written to disk before any final test evaluation."""

import argparse
import time

import numpy as np
import torch

from ship_detection.core import ROOT, TextScorer, dump, evaluate, prediction_cache, to_coco

ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0")
    p.add_argument("--text-device", default="cuda:0")
    p.add_argument("--models", nargs="+", choices=["n", "s"], default=["n", "s"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = p.parse_args()
    torch.set_num_threads(2)
    scorer = TextScorer(args.text_device)
    validation = []
    for kind in args.models:
        for seed in args.seeds:
            name = f"yolo11{kind}_seed{seed}"
            weights = ROOT / "runs" / name / "weights/best.pt"
            if not weights.exists():
                raise FileNotFoundError(weights)
            data = prediction_cache(
                weights,
                "val",
                args.device,
                ROOT / "artifacts/predictions" / f"{name}_val.npz",
                scorer,
            )
            row = {
                "run": name,
                "size": kind,
                "seed": seed,
                "baseline": evaluate(data, "val", details=True),
                "search": {},
            }
            for mode in ["names", "descriptions"]:
                grid = []
                for alpha in ALPHAS:
                    result = evaluate(data, "val", mode, alpha)
                    grid.append(result)
                # Deterministic tie-break: smaller text weight.
                best = max(grid, key=lambda x: (round(x["map"], 10), -x["alpha"]))
                row["search"][mode] = {
                    "grid": grid,
                    "selected_alpha": best["alpha"],
                    "selected_val_map": best["map"],
                }
            validation.append(row)
            dump(validation, ROOT / "artifacts/validation_results.json")
            print(
                "VALIDATION",
                name,
                row["baseline"]["map"],
                {m: v["selected_alpha"] for m, v in row["search"].items()},
                flush=True,
            )
    averages = {
        kind: float(np.mean([r["baseline"]["map"] for r in validation if r["size"] == kind]))
        for kind in args.models
    }
    selected_size = max(averages, key=averages.get)
    selection = {
        "selected_size": selected_size,
        "selection_rule": "highest mean internal-validation COCO mAP across requested seeds",
        "models": args.models,
        "seeds": args.seeds,
        "baseline_val_mean": averages,
        "alphas": ALPHAS,
        "confidence_threshold": 0.001,
        "nms_iou": 0.7,
        "precision_recall_confidence": 0.25,
        "nms_multi_label": False,
        "crop_margin": 0.05,
        "test_policy": "all settings frozen before opening final test predictions; no test-driven retuning",
        "predeclared_positive_weight_ablation": 0.25,
        "runs": {
            r["run"]: {m: r["search"][m]["selected_alpha"] for m in ["names", "descriptions"]}
            for r in validation
        },
        "frozen_at_unix": time.time(),
    }
    # Freeze all configurations BEFORE generating test predictions.
    dump(selection, ROOT / "artifacts/frozen_selection.json")
    print("FROZEN", selection, flush=True)
    test_results = []
    for row in validation:
        name = row["run"]
        weights = ROOT / "runs" / name / "weights/best.pt"
        # Both sizes are evaluated and disclosed; the primary size was chosen above.
        data = prediction_cache(
            weights,
            "test",
            args.device,
            ROOT / "artifacts/predictions" / f"{name}_test.npz",
            scorer,
        )
        methods = [
            ("baseline", 0.0),
            ("names", selection["runs"][name]["names"]),
            ("descriptions", selection["runs"][name]["descriptions"]),
        ]
        results = {
            mode: evaluate(data, "test", mode, alpha, details=True) for mode, alpha in methods
        }
        # Positive-weight ablation is predeclared, including when tuning chooses zero.
        results["descriptions_fixed025"] = evaluate(
            data, "test", "descriptions", 0.25, details=True
        )
        results["names_fixed025"] = evaluate(data, "test", "names", 0.25, details=True)
        result = {"run": name, "size": row["size"], "seed": row["seed"], "methods": results}
        test_results.append(result)
        dump(test_results, ROOT / "artifacts/test_results.json")
        for mode, alpha in methods:
            dump(
                to_coco(data, mode, alpha),
                ROOT / "artifacts/coco_predictions" / f"{name}_{mode}.json",
            )
        print("TEST", name, {m: round(r["map"], 4) for m, r in results.items()}, flush=True)
    summary = {}
    for kind in args.models:
        summary[kind] = {}
        for mode in test_results[0]["methods"]:
            rr = [r["methods"][mode] for r in test_results if r["size"] == kind]
            summary[kind][mode] = {
                metric: {
                    "mean": float(np.mean([r[metric] for r in rr])),
                    "std": float(np.std([r[metric] for r in rr], ddof=1 if len(rr) > 1 else 0)),
                }
                for metric in [
                    "map",
                    "map50",
                    "map75",
                    "precision_at_conf025",
                    "recall_at_conf025",
                    "matched_class_accuracy",
                ]
            }
    dump(summary, ROOT / "artifacts/aggregate_results.json")
    print("Evaluation complete", flush=True)


if __name__ == "__main__":
    main()
