"""Static attacker — uses original malicious prompt without mutation."""
import time
from typing import Dict, Any


def static_attack(
    model_adapter,
    ensemble_executor,
    ensemble_name: str,
    prompt: str,
    config: Dict[str, Any],
    judge_fn,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Static attacker: no mutation, just apply the original prompt directly.
    Returns dict with success_flag, final_prompt, used_queries, fail_layer, trajectory.
    """
    import time
    t0 = time.perf_counter()
    decision, transformed, trace = ensemble_executor.apply(ensemble_name, prompt, {})
    used_queries = 1

    traj_entry = {
        "prompt": prompt,
        "layer_decisions": trace,
        "response": None,
        "success_flag": False,
        "used_queries": used_queries,
        "fail_layer": None,
    }

    if decision == "BLOCK":
        fail_layer = _first_block_layer(trace)
        traj_entry["fail_layer"] = fail_layer
        return {
            "success": False,
            "final_prompt": prompt,
            "used_queries": used_queries,
            "fail_layer": fail_layer,
            "trajectory": [traj_entry],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    model_input = transformed or prompt
    response, lat, cost = model_adapter.generate(model_input, {})
    used_queries += 1
    vflag, vtype, jtrace = judge_fn(response, config)

    traj_entry["response"] = response[:200] if response else None
    traj_entry["success_flag"] = vflag is True
    traj_entry["used_queries"] = used_queries
    return {
        "success": vflag is True,
        "final_prompt": model_input,
        "used_queries": used_queries,
        "fail_layer": None,
        "trajectory": [traj_entry],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


def _first_block_layer(trace: list) -> str:
    for t in trace:
        if t["action"] == "BLOCK":
            return t["layer"]
    return "none"
