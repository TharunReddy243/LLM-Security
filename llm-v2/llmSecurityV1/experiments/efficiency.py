"""Efficiency analysis experiment — latency and cost statistics."""
from pathlib import Path
from typing import Dict, Any
import numpy as np


def run_efficiency_analysis(config: Dict[str, Any], logger) -> Dict[str, Any]:
    """
    RUN_EFFICIENCY_ANALYSIS: compute per-request latency and cost stats.
    """
    out_dir = Path(config["outputs"]["out_dir"])
    print("\n[EFFICIENCY ANALYSIS] Computing latency and cost stats...")

    events = logger.read_all()
    if not events:
        print("[WARN] No events to analyze.")
        return {}

    latencies = [float(ev["latency_ms"]) for ev in events if ev.get("latency_ms") is not None]
    costs = [float(ev.get("cost_estimate", 0.0)) for ev in events]

    def _stats(vals):
        if not vals:
            return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0, "count": 0}
        arr = np.array(vals)
        return {
            "mean": round(float(np.mean(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "count": len(arr),
        }

    lat_stats = _stats(latencies)
    cost_stats = _stats(costs)

    print(f"  Latency — Mean: {lat_stats['mean']}ms  P95: {lat_stats['p95']}ms  N={lat_stats['count']}")
    print(f"  Cost    — Mean: {cost_stats['mean']}  Total: {sum(costs):.4f}")

    results = {
        "mean_latency_ms": lat_stats["mean"],
        "p95_latency_ms": lat_stats["p95"],
        "latency_stats": lat_stats,
        "cost_stats": cost_stats,
    }

    # Save efficiency CSV
    import csv
    eff_path = out_dir / "analysis" / "efficiency.csv"
    eff_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eff_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in lat_stats.items():
            writer.writerow([f"latency_{k}", v])
        for k, v in cost_stats.items():
            writer.writerow([f"cost_{k}", v])
    print(f"  [CSV] Saved -> {eff_path}")

    return results
