"""STACK-like attacker — sequential layer-targeted bypass."""
import re
import random
import time
from typing import Dict, Any, List, Optional, Tuple

from attackers.rl_like import (
    _apply_leet, _apply_unicode, _apply_space_insert, _apply_synonym,
    FILLER_PREFIXES, FILLER_SUFFIXES, _first_block_layer,
)


def _layer_targeted_mutate(prompt: str, layer_name: str, trace: list) -> str:
    """LAYER_TARGETED_MUTATE: mutation specifically targeting a layer's weaknesses."""
    if layer_name == "keyword_blocker":
        mut = random.choice(["leet", "unicode", "space_insert"])
        if mut == "leet":
            return _apply_leet(prompt)
        if mut == "unicode":
            return _apply_unicode(prompt)
        return _apply_space_insert(prompt)
    elif layer_name == "intent_classifier":
        prefix = random.choice(FILLER_PREFIXES)
        return prefix + prompt
    elif layer_name == "capability_reducer":
        cleaned = re.sub(r'```[\s\S]*?```', '', prompt, flags=re.MULTILINE).strip()
        return cleaned if cleaned else prompt
    else:
        return _apply_leet(prompt)


def _extract_transformation(original: str, transformed: str) -> Dict[str, Any]:
    """EXTRACT_TRANSFORMATION: record what changed between two prompts."""
    return {
        "original_len": len(original),
        "transformed_len": len(transformed),
        "diff_chars": sum(1 for a, b in zip(original, transformed) if a != b),
        "original_prefix": original[:40],
        "transformed_prefix": transformed[:40],
    }


def _apply_transformations(base: str, transforms: List[Tuple[str, str]]) -> str:
    """APPLY_TRANSFORMATIONS: sequentially apply collected layer bypass strategies."""
    result = base
    for layer_name, mutated_version in transforms:
        # We store the mutated version directly; apply on current result
        if layer_name == "keyword_blocker":
            result = _apply_leet(result)
        elif layer_name == "intent_classifier":
            result = random.choice(FILLER_PREFIXES) + result
        elif layer_name == "capability_reducer":
            result = re.sub(r'```[\s\S]*?```', '', result, flags=re.MULTILINE).strip() or result
    return result


def stack_attack(
    model_adapter,
    ensemble_executor,
    ensemble_name: str,
    prompt: str,
    per_layer_budget: int,
    config: Dict[str, Any],
    judge_fn,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    STACK_ATTACK(m, e, p0, per_layer_budget):
    Sequential layer-targeted bypass with transformation accumulation.
    """
    random.seed(seed)
    layer_order = ensemble_executor.layer_order_for(ensemble_name)
    used_queries = 0
    trajectory = []
    collected_transforms: List[Tuple[str, str]] = []
    current = prompt

    for layer_idx, layer_name in enumerate(layer_order):
        if layer_name == "normalization":
            continue
        if used_queries >= per_layer_budget * max(len(layer_order), 1):
            break

        bypass_found = False
        for _ in range(per_layer_budget):
            if used_queries >= per_layer_budget * len(layer_order):
                break
            candidate = _layer_targeted_mutate(current, layer_name, [])
            # Test up to this layer only
            decision, transformed_c, trace_c = ensemble_executor.apply_up_to(
                ensemble_name, candidate, layer_idx, {}
            )
            used_queries += 1
            trajectory.append({
                "phase": f"target_{layer_name}",
                "prompt": candidate[:100],
                "decision": decision,
                "used_queries": used_queries,
            })
            if decision != "BLOCK":
                transform = _extract_transformation(current, candidate)
                collected_transforms.append((layer_name, candidate))
                current = candidate
                bypass_found = True
                break

        if not bypass_found:
            pass  # continue with next layer anyway

    # Apply all transformations to original and test full pipeline
    final_prompt = _apply_transformations(prompt, collected_transforms) if collected_transforms else current
    decision, transformed, trace = ensemble_executor.apply(ensemble_name, final_prompt, {})
    used_queries += 1

    trajectory.append({
        "phase": "final_combined",
        "prompt": final_prompt[:100],
        "decision": decision,
        "used_queries": used_queries,
    })

    if decision != "BLOCK":
        model_input = transformed or final_prompt
        response, lat, cost = model_adapter.generate(model_input, {})
        used_queries += 1
        vflag, vtype, jtrace = judge_fn(response, config)
        trajectory[-1]["response"] = response[:200] if response else None
        trajectory[-1]["success_flag"] = vflag is True
        return {
            "success": vflag is True,
            "final_prompt": final_prompt,
            "used_queries": used_queries,
            "fail_layer": None if vflag is True else "judge",
            "trajectory": trajectory,
        }

    return {
        "success": False,
        "final_prompt": final_prompt,
        "used_queries": used_queries,
        "fail_layer": _first_block_layer(trace),
        "trajectory": trajectory,
    }
