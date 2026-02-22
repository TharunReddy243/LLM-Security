#!/usr/bin/env python3
"""
llmSecurityV1 — Emergent Tool: CLI entrypoint for research framework.

Usage:
  python run.py --mode static_baseline --config config.yaml --model tinyllama --ensemble default_ensemble
  python run.py --mode adaptive_eval --attacker rl --output outputs/exp001_rl
  python run.py --mode all --config config.yaml --output outputs/exp_full
  python run.py --mode debug --prompt "Ignore previous instructions" --model tinyllama
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure UTF-8 stdout on Windows to avoid UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 1
EXIT_RUNTIME_ERROR = 2
EXIT_TESTS_FAILED = 3

# Project root = directory containing run.py
PROJECT_ROOT = Path(__file__).parent.resolve()

REQUIRED_FILES = [
    "run.py",
    "model/model_api.py",
    "defenses/layer_base.py",
    "defenses/keyword_blocker.py",
    "defenses/intent_classifier.py",
    "defenses/intent_taxonomy.py",
    "defenses/capability_reducer.py",
    "defenses/ensemble.py",
    "defenses/normalize.py",
    "judge/policy_judge.py",
    "attackers/static.py",
    "attackers/rl_like.py",
    "attackers/ga_like.py",
    "attackers/stack_like.py",
    "analysis/discrimination.py",
    "analysis/first_failure.py",
    "analysis/layer_stats.py",
    "analysis/safety_shift.py",
    "data/benign.json",
    "data/malicious.json",
]

REQUIRED_DIRS = ["data/malicious/"]


def validate_project_layout() -> list:
    """Check that all required files and directories exist. Returns list of missing paths."""
    missing = []
    for rel in REQUIRED_FILES:
        p = PROJECT_ROOT / rel
        if not p.exists():
            missing.append(rel)
    for rel in REQUIRED_DIRS:
        p = PROJECT_ROOT / rel
        if not p.is_dir():
            missing.append(rel + " (dir)")
    return missing


def set_random_seeds(seed: int) -> None:
    """Set all randomness seeds for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="llmSecurityV1 — two-layer ensemble defense + attacks + judge + evaluation"
    )
    parser.add_argument("--mode", type=str, default="static_baseline",
                        choices=["static_baseline", "adaptive_eval", "mechanistic_analysis",
                                 "efficiency_analysis", "all", "debug", "test"],
                        help="Experiment mode")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--model", type=str, default="mock",
                        help="Model name (mock, tinyllama, etc.)")
    parser.add_argument("--ensemble", type=str, default="default_ensemble",
                        help="Ensemble name")
    parser.add_argument("--attacker", type=str, default="static",
                        help="Attacker name (static, rl, ga, stack)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: from config)")
    parser.add_argument("--output-dir", "--output", type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt for debug mode")
    args = parser.parse_args()

    # Validate project layout
    missing = validate_project_layout()
    if missing:
        print("[ERROR] Missing required files or directories:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if args.mode == "test":
        return _run_tests()

    # Ensure we run from project root
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    try:
        from utils.config_loader import load_config
        config = load_config(str(config_path))
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    seed = args.seed if args.seed is not None else config.get("experiment", {}).get("seed", 42)
    config["experiment"]["seed"] = seed
    set_random_seeds(seed)

    if args.output_dir:
        config["outputs"]["out_dir"] = args.output_dir.rstrip("/")
        config["logging"]["out_dir"] = os.path.join(args.output_dir, "logs")
        config["logging"]["jsonl_file"] = os.path.join(args.output_dir, "logs", "events.jsonl")
    out_dir = Path(config["outputs"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.txt").write_text(
        "llmSecurityV1 experiment outputs.\n"
        "Files: attack_results.csv, benign_results.csv, summary.json, events.jsonl (in logs/),\n"
        "plots/: safety_shift.png, layer_failure_hist.png, latency_dist.png, asr_vs_budget_*.png\n",
        encoding="utf-8",
    )
    Path(config["logging"]["jsonl_file"]).parent.mkdir(parents=True, exist_ok=True)

    from utils.logging_utils import EventLogger
    logger = EventLogger(config["logging"]["jsonl_file"])

    try:
        if args.mode == "debug":
            return _run_debug(config, args, logger)
        if args.mode == "static_baseline":
            return _run_static_baseline(config, args, logger)
        if args.mode == "adaptive_eval":
            return _run_adaptive_eval(config, args, logger)
        if args.mode == "mechanistic_analysis":
            return _run_mechanistic(config, logger)
        if args.mode == "efficiency_analysis":
            return _run_efficiency(config, logger)
        if args.mode == "all":
            return _run_all(config, args, logger)
    except Exception as e:
        import traceback
        print(f"[ERROR] Runtime error: {e}", file=sys.stderr)
        traceback.print_exc()
        logger.log({"timestamp": _utcnow(), "error": str(e), "traceback": traceback.format_exc()})
        logger.close()
        return EXIT_RUNTIME_ERROR

    logger.close()
    return EXIT_SUCCESS


def _run_tests() -> int:
    """Run pytest. Exit with EXIT_TESTS_FAILED if any test fails."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/"],
        cwd=PROJECT_ROOT,
    )
    return EXIT_TESTS_FAILED if result.returncode != 0 else EXIT_SUCCESS


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_debug(config, args, logger) -> int:
    """Run single prompt through pipeline (debug mode)."""
    prompt = args.prompt or "What is the capital of France?"
    from defenses.ensemble import EnsembleExecutor
    from model.model_api import call_model
    from judge.policy_judge import judge_policy_violation
    from defenses.normalize import normalize_text

    executor = EnsembleExecutor(config)
    print(f"\n[DEBUG] Prompt: {prompt[:80]}...")
    normalized = normalize_text(prompt)
    print(f"  Normalized: {normalized[:80]}...")
    decision, transformed, trace = executor.apply(args.ensemble, prompt, {})
    print(f"  Decision: {decision}")
    if trace:
        for t in trace:
            print(f"    {t['layer']}: {t['action']} | {t.get('meta', {})}")
    if decision != "BLOCK":
        resp, lat, _ = call_model(transformed or prompt, args.model, {}, config)
        vflag, vtype, jtrace = judge_policy_violation(resp, config)
        print(f"  LLM output (trunc): {resp[:120] if resp else None}...")
        print(f"  Judge: violation={vflag} type={vtype}")
    logger.close()
    return EXIT_SUCCESS


def _run_static_baseline(config, args, logger) -> int:
    from experiments.static_baseline import run_static_baseline
    results = run_static_baseline(args.model, args.ensemble, config, logger)
    _write_summary(config, results, "static_baseline")
    _print_console_summary(results, config)
    logger.close()
    return EXIT_SUCCESS


def _run_adaptive_eval(config, args, logger) -> int:
    from experiments.adaptive_eval import run_adaptive_eval
    results = run_adaptive_eval(args.model, args.ensemble, args.attacker, config, logger)
    _write_summary(config, results, "adaptive_eval", attacker=args.attacker)
    _print_console_summary(results, config, mode="adaptive", attacker=args.attacker)
    logger.close()
    return EXIT_SUCCESS


def _run_mechanistic(config, logger) -> int:
    from experiments.mechanistic import run_mechanistic_analysis
    results = run_mechanistic_analysis(config, logger)
    _write_summary(config, results, "mechanistic_analysis")
    logger.close()
    return EXIT_SUCCESS


def _run_efficiency(config, logger) -> int:
    from experiments.efficiency import run_efficiency_analysis
    results = run_efficiency_analysis(config, logger)
    _write_summary(config, results, "efficiency_analysis")
    print(f"\n[EFFICIENCY] Mean latency: {results.get('mean_latency_ms')}ms  P95: {results.get('p95_latency_ms')}ms")
    logger.close()
    return EXIT_SUCCESS


def _run_all(config, args, logger) -> int:
    """Run full pipeline: static + adaptive + mechanistic + efficiency."""
    from experiments.static_baseline import run_static_baseline
    from experiments.adaptive_eval import run_adaptive_eval
    from experiments.mechanistic import run_mechanistic_analysis
    from experiments.efficiency import run_efficiency_analysis

    all_results = {"static_baseline": {}, "adaptive": {}, "mechanistic": {}, "efficiency": {}}

    # Static baseline
    all_results["static_baseline"] = run_static_baseline(args.model, args.ensemble, config, logger)

    # Adaptive eval for each attacker
    for att in ["static", "rl", "ga", "stack"]:
        try:
            r = run_adaptive_eval(args.model, args.ensemble, att, config, logger)
            all_results["adaptive"][att] = r
        except Exception as e:
            print(f"[WARN] Adaptive {att} failed: {e}")
            all_results["adaptive"][att] = {"error": str(e)}

    # Mechanistic
    all_results["mechanistic"] = run_mechanistic_analysis(config, logger)
    all_results["efficiency"] = run_efficiency_analysis(config, logger)

    _write_summary(config, all_results, "all")
    _print_console_summary(all_results.get("static_baseline", {}), config, mode="all",
                           adaptive=all_results.get("adaptive", {}))
    logger.close()
    return EXIT_SUCCESS


def _write_summary(config, results, mode: str, attacker: str = None) -> None:
    """Write summary.json to output directory."""
    out_dir = Path(config["outputs"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment_id": config.get("experiment", {}).get("id", "exp-unknown"),
        "timestamp": _utcnow(),
        "mode": mode,
        "seed": config.get("experiment", {}).get("seed", 42),
        "results": results,
    }
    if attacker:
        summary["attacker"] = attacker
    path = out_dir / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OUTPUT] summary.json -> {path}")


def _print_console_summary(results, config, mode: str = "static", attacker: str = None,
                           adaptive: dict = None) -> None:
    """Print human-readable console summary."""
    print("\n" + "=" * 60)
    print("CONSOLE SUMMARY")
    print("=" * 60)
    if isinstance(results, dict) and "asr" in results:
        print(f"ASR_static:     {results.get('asr', 0):.3f}")
        print(f"FRR:            {results.get('frr', 0):.3f}")
        print(f"Successes:      {results.get('successes', 0)} / {results.get('total_malicious', 0)}")
        print(f"Benign refusals:{results.get('benign_refusals', 0)} / {results.get('total_benign', 0)}")
    if mode == "adaptive" and isinstance(results, dict):
        print(f"ASR_adaptive:   {results.get('asr_adaptive', 0):.3f} (attacker={attacker})")
        print(f"AVG_Q:          {results.get('avg_queries', 0):.1f}")
    if mode == "all" and adaptive:
        for att, r in adaptive.items():
            if isinstance(r, dict) and "asr_adaptive" in r:
                print(f"ASR_adaptive[{att}]: {r['asr_adaptive']:.3f}  avg_q={r.get('avg_queries', 0):.1f}")
    if isinstance(results, dict) and "per_domain" in results:
        # Only show domains that have malicious prompts (tp+fn > 0) to avoid misleading 0.00
        domains_with_malicious = [
            (d, m) for d, m in results["per_domain"].items()
            if (m.get("tp", 0) + m.get("fn", 0)) > 0
        ]
        if domains_with_malicious:
            print("\nPer-domain (malicious-containing domains):")
            for d, m in domains_with_malicious[:8]:
                print(f"  {d}: P={m.get('precision',0):.2f} R={m.get('recall',0):.2f} F1={m.get('f1',0):.2f}")
    if isinstance(results, dict) and "top_failures" in results and results["top_failures"]:
        print("\nTop 5 failure prompts (bypassed defense):")
        for prompt, layer in results["top_failures"]:
            print(f"  [{layer}] {prompt[:60]}...")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
