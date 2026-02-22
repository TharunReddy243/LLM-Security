"""Mechanistic analysis experiment."""
import json
from pathlib import Path
from typing import Dict, Any

from analysis.discrimination import compute_discrimination_stats, save_discrimination_csv
from analysis.first_failure import compute_first_failure_stats, save_first_failure_csv
from analysis.layer_stats import compute_layer_stats, save_layer_stats_csv
from analysis.safety_shift import compute_safety_shift, save_safety_shift_csv


def run_mechanistic_analysis(config: Dict[str, Any], logger) -> Dict[str, Any]:
    """
    RUN_MECHANISTIC_ANALYSIS: compute discrimination, failure, layer stats from logs.
    """
    out_dir = Path(config["outputs"]["out_dir"])
    log_path = Path(config["logging"]["jsonl_file"])
    print(f"\n[MECHANISTIC ANALYSIS] Loading events from {log_path}...")

    events = logger.read_all()
    if not events:
        print("[WARN] No events to analyze. Run static_baseline first.")
        return {}

    print(f"  Loaded {len(events)} events.")

    # Discrimination stats
    disc_stats = compute_discrimination_stats(events)
    save_discrimination_csv(disc_stats, str(out_dir))
    print(f"  [DISC] AUC={disc_stats.get('auc')}  KS={disc_stats.get('ks_statistic')}")

    # First failure stats
    ff_stats = compute_first_failure_stats(events)
    save_first_failure_csv(ff_stats, str(out_dir))
    print(f"  [FAILURE] Layer failure counts: {ff_stats}")

    # Layer latency stats
    lat_stats = compute_layer_stats(events)
    save_layer_stats_csv(lat_stats, str(out_dir))
    e2e = lat_stats.get("end_to_end", {})
    print(f"  [LATENCY] Mean={e2e.get('mean')}ms  P95={e2e.get('p95')}ms")

    # Safety shift (use same events for pre/post as approximation if no post available)
    shift_data = compute_safety_shift(events, events)
    save_safety_shift_csv(shift_data, str(out_dir))

    # Plots
    if config.get("evaluation", {}).get("save_plots", True):
        try:
            from utils.plotting import save_layer_failure_hist, save_latency_dist, save_safety_shift
            if ff_stats:
                p = save_layer_failure_hist(ff_stats, str(out_dir))
                print(f"  [PLOT] {p}")
            latencies = [ev.get("latency_ms", 0) for ev in events if ev.get("latency_ms")]
            if latencies:
                p = save_latency_dist(latencies, str(out_dir))
                print(f"  [PLOT] {p}")
            if shift_data.get("pre_frr") and shift_data.get("post_frr"):
                p = save_safety_shift(shift_data["pre_frr"], shift_data["post_frr"], str(out_dir))
                print(f"  [PLOT] {p}")
        except Exception as e:
            print(f"  [WARN] Plot error: {e}")

    results = {
        "discrimination": disc_stats,
        "first_failure": ff_stats,
        "latency": lat_stats,
        "safety_shift": shift_data,
    }

    # Save mechanistic CSV summary
    mech_path = out_dir / "analysis" / "mechanistic_analysis.csv"
    mech_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(mech_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["auc", disc_stats.get("auc", "N/A")])
        writer.writerow(["ks_statistic", disc_stats.get("ks_statistic", "N/A")])
        writer.writerow(["mean_latency_ms", e2e.get("mean", 0)])
        writer.writerow(["p95_latency_ms", e2e.get("p95", 0)])
        for layer, cnt in ff_stats.items():
            writer.writerow([f"first_failure_{layer}", cnt])

    print(f"  [CSV] Saved -> {mech_path}")
    return results
