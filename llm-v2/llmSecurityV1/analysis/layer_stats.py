"""Layer statistics — per-layer and end-to-end latency computation."""
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


def compute_layer_stats(events: List[Dict]) -> Dict[str, Any]:
    """Compute per-layer and end-to-end latency statistics."""
    per_layer: Dict[str, List[float]] = defaultdict(list)
    end_to_end: List[float] = []

    for ev in events:
        lat = ev.get("latency_ms")
        if lat is not None:
            end_to_end.append(float(lat))
        for t in ev.get("layer_trace", []):
            layer_lat = t.get("latency_ms")
            if layer_lat is not None:
                per_layer[t["layer"]].append(float(layer_lat))

    def _stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "count": 0}
        arr = np.array(vals)
        return {
            "mean": round(float(np.mean(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "count": len(arr),
        }

    return {
        "end_to_end": _stats(end_to_end),
        "per_layer": {layer: _stats(vals) for layer, vals in per_layer.items()},
    }


def save_layer_stats_csv(stats: Dict[str, Any], out_dir: str) -> str:
    path = Path(out_dir) / "analysis" / "layer_stats.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "mean_ms", "median_ms", "p95_ms", "count"])
        e2e = stats.get("end_to_end", {})
        writer.writerow(["end_to_end", e2e.get("mean", 0), e2e.get("median", 0), e2e.get("p95", 0), e2e.get("count", 0)])
        for layer, s in stats.get("per_layer", {}).items():
            writer.writerow([layer, s.get("mean", 0), s.get("median", 0), s.get("p95", 0), s.get("count", 0)])
    return str(path)
