"""Discrimination analysis — AUC and KS statistic between malicious and benign scores."""
import json
import csv
from pathlib import Path
from typing import Dict, Any, List


def compute_discrimination_stats(events: List[Dict]) -> Dict[str, Any]:
    """Compute AUC and KS statistic from logged events."""
    mal_scores, ben_scores = [], []
    for ev in events:
        trace = ev.get("layer_trace", [])
        max_score = 0.0
        for t in trace:
            s = t.get("meta", {}).get("score", 0.0)
            if s and s > max_score:
                max_score = float(s)
        prompt_id = ev.get("prompt_id", "")
        if "ben" in prompt_id.lower():
            ben_scores.append(max_score)
        elif "mal" in prompt_id.lower() or "vio" in prompt_id.lower() or "drug" in prompt_id.lower():
            mal_scores.append(max_score)

    stats: Dict[str, Any] = {
        "n_malicious": len(mal_scores),
        "n_benign": len(ben_scores),
        "auc": None,
        "ks_statistic": None,
        "mean_malicious_score": round(sum(mal_scores) / len(mal_scores), 4) if mal_scores else 0.0,
        "mean_benign_score": round(sum(ben_scores) / len(ben_scores), 4) if ben_scores else 0.0,
    }

    if mal_scores and ben_scores:
        try:
            from sklearn.metrics import roc_auc_score
            import numpy as np
            y_true = [1] * len(mal_scores) + [0] * len(ben_scores)
            y_score = mal_scores + ben_scores
            stats["auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        except Exception:
            pass
        try:
            from scipy.stats import ks_2samp
            ks_stat, ks_p = ks_2samp(mal_scores, ben_scores)
            stats["ks_statistic"] = round(float(ks_stat), 4)
            stats["ks_pvalue"] = round(float(ks_p), 4)
        except Exception:
            pass

    return stats


def save_discrimination_csv(stats: Dict[str, Any], out_dir: str) -> str:
    path = Path(out_dir) / "analysis" / "discrimination_stats.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
        writer.writeheader()
        writer.writerow(stats)
    return str(path)
