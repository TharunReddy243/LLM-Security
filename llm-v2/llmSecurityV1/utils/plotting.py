"""Matplotlib-only plotting utilities."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


def save_safety_shift(pre_frr: Dict[str, float], post_frr: Dict[str, float], out_dir: str) -> str:
    domains = sorted(set(list(pre_frr.keys()) + list(post_frr.keys())))
    x = np.arange(len(domains))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    pre_vals = [pre_frr.get(d, 0.0) for d in domains]
    post_vals = [post_frr.get(d, 0.0) for d in domains]
    ax.bar(x - width / 2, pre_vals, width, label="Pre-attack FRR")
    ax.bar(x + width / 2, post_vals, width, label="Post-attack FRR")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_ylabel("False Refusal Rate")
    ax.set_title("Safety Shift: Benign Refusal Rate Pre vs Post Attack")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = Path(out_dir) / "plots" / "safety_shift.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=100)
    plt.close(fig)
    return str(path)


def save_layer_failure_hist(layer_failure_counts: Dict[str, int], out_dir: str) -> str:
    layers = list(layer_failure_counts.keys())
    counts = [layer_failure_counts[l] for l in layers]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(layers, counts)
    ax.set_xlabel("Layer")
    ax.set_ylabel("First-Failure Count")
    ax.set_title("Layer Failure Histogram (First Blocking Layer)")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.1, str(v), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = Path(out_dir) / "plots" / "layer_failure_hist.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=100)
    plt.close(fig)
    return str(path)


def save_latency_dist(latencies_ms: List[float], out_dir: str) -> str:
    if not latencies_ms:
        return ""
    arr = np.array(latencies_ms)
    mean_v = float(np.mean(arr))
    median_v = float(np.median(arr))
    p95_v = float(np.percentile(arr, 95))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(arr, bins=min(30, len(arr)), edgecolor="black")
    ax.axvline(mean_v, linestyle="--", label=f"Mean={mean_v:.1f}ms")
    ax.axvline(median_v, linestyle=":", label=f"Median={median_v:.1f}ms")
    ax.axvline(p95_v, linestyle="-.", label=f"P95={p95_v:.1f}ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("End-to-End Latency Distribution")
    ax.legend()
    fig.tight_layout()
    path = Path(out_dir) / "plots" / "latency_dist.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=100)
    plt.close(fig)
    return str(path)


def save_asr_vs_budget(budget_asr: Dict[int, float], attacker_name: str, out_dir: str) -> str:
    if not budget_asr:
        return ""
    budgets = sorted(budget_asr.keys())
    asrs = [budget_asr[b] for b in budgets]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(budgets, asrs, marker="o")
    ax.set_xlabel("Query Budget")
    ax.set_ylabel("Attack Success Rate (ASR)")
    ax.set_title(f"ASR vs Query Budget ({attacker_name})")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = Path(out_dir) / "plots" / f"asr_vs_budget_{attacker_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=100)
    plt.close(fig)
    return str(path)
