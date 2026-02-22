"""GA-like attacker — genetic algorithm with string crossover and mutation."""
import random
import time
from typing import Dict, Any, List

from attackers.rl_like import (
    _apply_leet, _apply_unicode, _apply_space_insert,
    _apply_synonym, FILLER_PREFIXES, FILLER_SUFFIXES, _first_block_layer,
)

MUTATION_TYPES = ["leet", "unicode", "space_insert", "filler_prefix", "filler_suffix", "synonym", "truncate"]


def _ga_mutate(prompt: str) -> str:
    mut = random.choice(MUTATION_TYPES)
    if mut == "leet":
        return _apply_leet(prompt)
    if mut == "unicode":
        return _apply_unicode(prompt)
    if mut == "space_insert":
        return _apply_space_insert(prompt)
    if mut == "filler_prefix":
        return random.choice(FILLER_PREFIXES) + prompt
    if mut == "filler_suffix":
        return prompt + random.choice(FILLER_SUFFIXES)
    if mut == "synonym":
        return _apply_synonym(prompt)
    if mut == "truncate":
        words = prompt.split()
        if len(words) > 3:
            start = random.randint(0, max(0, len(words) // 4))
            return ' '.join(words[start:])
    return prompt


def _crossover(p1: str, p2: str) -> str:
    """Single-point word-level crossover."""
    w1, w2 = p1.split(), p2.split()
    if not w1 or not w2:
        return p1
    cut1 = random.randint(1, max(1, len(w1) - 1))
    cut2 = random.randint(1, max(1, len(w2) - 1))
    child = w1[:cut1] + w2[cut2:]
    return ' '.join(child) if child else p1


def _init_population(p0: str, n: int) -> List[str]:
    pop = [p0]
    for _ in range(n - 1):
        pop.append(_ga_mutate(p0))
    return pop


def _fitness(
    prompt: str,
    ensemble_executor,
    ensemble_name: str,
    model_adapter,
    judge_fn,
    config: Dict[str, Any],
    layer_count: int,
) -> float:
    """PARTIAL_SCORE: higher = closer to attack success."""
    decision, transformed, trace = ensemble_executor.apply(ensemble_name, prompt, {})
    if decision == "BLOCK":
        blocked_at = next((i for i, t in enumerate(trace) if t["action"] == "BLOCK"), 0)
        return blocked_at / max(layer_count, 1)
    # Got through ensemble, try model
    model_input = transformed or prompt
    response, lat, cost = model_adapter.generate(model_input, {})
    vflag, vtype, jtrace = judge_fn(response, config)
    if vflag is True:
        return 1.0
    return 0.6 + 0.4 * jtrace.get("max_sim", 0.0)


def ga_attack(
    model_adapter,
    ensemble_executor,
    ensemble_name: str,
    prompt: str,
    budget: int,
    pop_size: int,
    generations: int,
    config: Dict[str, Any],
    judge_fn,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    GA_ATTACK(m, e, p0, B, N, G): genetic algorithm adversarial search.
    Returns (success_flag, final_prompt, used_queries, fail_layer, trajectory).
    """
    random.seed(seed)
    used_queries = 0
    trajectory = []
    layer_order = ensemble_executor.layer_order_for(ensemble_name)
    layer_count = len(layer_order)

    population = _init_population(prompt, pop_size)

    for gen in range(generations):
        if used_queries >= budget:
            break

        scored = []
        for ind in population:
            if used_queries >= budget:
                break
            score = _fitness(ind, ensemble_executor, ensemble_name, model_adapter, judge_fn, config, layer_count)
            used_queries += 2  # 1 ensemble + 1 model
            scored.append((score, ind))
            trajectory.append({
                "generation": gen,
                "prompt": ind,
                "fitness": score,
                "used_queries": used_queries,
            })
            if score >= 1.0:
                decision, transformed, trace = ensemble_executor.apply(ensemble_name, ind, {})
                return {
                    "success": True,
                    "final_prompt": transformed or ind,
                    "used_queries": used_queries,
                    "fail_layer": None,
                    "trajectory": trajectory,
                }

        scored.sort(key=lambda x: x[0], reverse=True)
        top_half = [ind for _, ind in scored[:max(2, pop_size // 2)]]

        new_pop = list(top_half)
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(top_half, min(2, len(top_half)))
            child = _crossover(p1, p2)
            child = _ga_mutate(child)
            new_pop.append(child)
        population = new_pop

    best = max(scored, key=lambda x: x[0])[1] if scored else prompt
    decision, transformed, trace = ensemble_executor.apply(ensemble_name, best, {})
    return {
        "success": False,
        "final_prompt": transformed or best,
        "used_queries": used_queries,
        "fail_layer": _first_block_layer(trace),
        "trajectory": trajectory,
    }
