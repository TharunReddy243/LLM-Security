"""Adaptive evaluation experiment — runs attackers against the ensemble."""
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from defenses.ensemble import EnsembleExecutor
from model.model_api import get_model_adapter
from judge.policy_judge import judge_policy_violation


def _load_prompts(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Data file not found: {path}")
        return []
    with open(p) as f:
        return json.load(f)


def _get_attacker(attacker_name: str, config: Dict[str, Any]):
    atk_cfg = config.get("attackers", {}).get(attacker_name, {})
    if attacker_name == "rl":
        from attackers.rl_like import rl_attack
        budget = atk_cfg.get("budget", 100)
        return lambda adapter, executor, ensemble, prompt, seed: rl_attack(
            adapter, executor, ensemble, prompt, budget, config, judge_policy_violation, seed
        )
    elif attacker_name == "ga":
        from attackers.ga_like import ga_attack
        budget = atk_cfg.get("budget", 200)
        pop_size = atk_cfg.get("pop_size", 20)
        generations = atk_cfg.get("generations", 10)
        return lambda adapter, executor, ensemble, prompt, seed: ga_attack(
            adapter, executor, ensemble, prompt, budget, pop_size, generations, config, judge_policy_violation, seed
        )
    elif attacker_name == "stack":
        from attackers.stack_like import stack_attack
        per_layer_budget = atk_cfg.get("per_layer_budget", 20)
        return lambda adapter, executor, ensemble, prompt, seed: stack_attack(
            adapter, executor, ensemble, prompt, per_layer_budget, config, judge_policy_violation, seed
        )
    elif attacker_name == "static":
        from attackers.static import static_attack
        return lambda adapter, executor, ensemble, prompt, seed: static_attack(
            adapter, executor, ensemble, prompt, config, judge_policy_violation, seed
        )
    raise ValueError(f"Unknown attacker: {attacker_name!r}")


def run_adaptive_eval(
    model_name: str,
    ensemble_name: str,
    attacker_name: str,
    config: Dict[str, Any],
    logger,
) -> Dict[str, Any]:
    """
    RUN_ADAPTIVE_EVAL: run attacker on each seed in Pm, compute ASR_adaptive and AVG_Q.
    """
    pm_path = config["data"]["malicious"]
    out_dir = Path(config["outputs"]["out_dir"])
    exp_id = config["experiment"]["id"]
    base_seed = config["experiment"]["seed"]

    Pm = _load_prompts(pm_path)
    if not Pm:
        print("[WARN] No malicious prompts loaded for adaptive eval.")
        return {"asr_adaptive": 0.0, "avg_queries": 0, "total": 0}

    adapter = get_model_adapter(model_name, config)
    executor = EnsembleExecutor(config)
    attacker_fn = _get_attacker(attacker_name, config)

    print(f"\n[ADAPTIVE EVAL] Attacker={attacker_name}, Model={model_name}, {len(Pm)} seeds...")

    success_count = 0
    total_queries = 0
    all_trajectories = []

    for i, pm in enumerate(Pm):
        text = pm["text"]
        pid = pm.get("id", f"mal-{i:04d}")
        seed = base_seed + i

        t0 = time.perf_counter()
        result = attacker_fn(adapter, executor, ensemble_name, text, seed)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)

        qused = result["used_queries"]
        total_queries += qused

        if result["success"]:
            success_count += 1

        all_trajectories.append({
            "prompt_id": pid,
            "seed_prompt": text[:100],
            "success": result["success"],
            "used_queries": qused,
            "fail_layer": result.get("fail_layer"),
            "final_prompt": result.get("final_prompt", "")[:100],
            "trajectory_len": len(result.get("trajectory", [])),
            "latency_ms": elapsed,
        })

        logger.log({
            "experiment_id": exp_id,
            "event": "adaptive_attack",
            "attacker": attacker_name,
            "model": model_name,
            "ensemble": ensemble_name,
            "prompt_id": pid,
            "seed_prompt": text[:100],
            "success": result["success"],
            "used_queries": qused,
            "fail_layer": result.get("fail_layer"),
            "trajectory_len": len(result.get("trajectory", [])),
            "latency_ms": elapsed,
        })

        status = "SUCCESS" if result["success"] else "FAIL"
        print(f"  [{i+1}/{len(Pm)}] {pid}: {status} | queries={qused} | fail_layer={result.get('fail_layer')}")

    asr_adaptive = round(success_count / len(Pm), 4)
    avg_queries = round(total_queries / len(Pm), 2)

    # Budget-ASR curve data
    budget_steps = {}
    for traj in all_trajectories:
        q = traj["used_queries"]
        if traj["success"]:
            for bstep in range(1, q + 1):
                budget_steps[bstep] = budget_steps.get(bstep, 0) + 1

    max_budget = max((t["used_queries"] for t in all_trajectories), default=1)
    budget_asr = {}
    total = len(Pm)
    for b in range(1, min(max_budget + 1, 201)):
        succ = sum(1 for t in all_trajectories if t["success"] and t["used_queries"] <= b)
        budget_asr[b] = round(succ / total, 4)

    # Save trajectory JSONL
    traj_path = out_dir / f"trajectories_{attacker_name}.jsonl"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    with open(traj_path, "w") as f:
        for t in all_trajectories:
            f.write(json.dumps(t) + "\n")

    results = {
        "attacker": attacker_name,
        "asr_adaptive": asr_adaptive,
        "avg_queries": avg_queries,
        "total": len(Pm),
        "successes": success_count,
        "budget_asr": budget_asr,
    }

    print(f"\n[ADAPTIVE EVAL] ASR_adaptive={asr_adaptive:.3f}  AVG_Q={avg_queries:.1f}")

    # Plots
    if config.get("evaluation", {}).get("save_plots", True):
        try:
            from utils.plotting import save_asr_vs_budget
            p = save_asr_vs_budget(budget_asr, attacker_name, str(out_dir))
            print(f"  [PLOT] {p}")
        except Exception as e:
            print(f"  [WARN] Plot failed: {e}")

    return results
