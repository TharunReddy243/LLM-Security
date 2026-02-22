"""Static baseline experiment."""
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from defenses.ensemble import EnsembleExecutor, apply_ensemble
from model.model_api import call_model
from judge.policy_judge import judge_policy_violation
from defenses.normalize import normalize_text


def _load_prompts(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Data file not found: {path}")
        return []
    with open(p) as f:
        return json.load(f)


def _first_block_layer(trace: list) -> str:
    for t in trace:
        if t.get("action") == "BLOCK":
            return t["layer"]
    return "none"


def run_static_baseline(
    model_name: str,
    ensemble_name: str,
    config: Dict[str, Any],
    logger,
) -> Dict[str, Any]:
    """
    RUN_STATIC_BASELINE: evaluate ensemble on malicious (Pm) and benign (Pb) prompts.
    Computes ASR_static and FRR. Saves per-prompt CSV.
    """
    pm_path = config["data"]["benign"].replace("benign", "malicious")
    pm_path = config["data"]["malicious"]
    pb_path = config["data"]["benign"]
    out_dir = Path(config["outputs"]["out_dir"])
    exp_id = config["experiment"]["id"]
    seed = config["experiment"]["seed"]

    Pm = _load_prompts(pm_path)
    Pb = _load_prompts(pb_path)

    if not Pm:
        print("[WARN] No malicious prompts loaded.")
    if not Pb:
        print("[WARN] No benign prompts loaded.")

    executor = EnsembleExecutor(config)
    attack_rows, benign_rows = [], []
    success_count = 0
    benign_refusal_count = 0

    # Per-domain tracking
    domain_tp: Dict[str, int] = {}
    domain_fp: Dict[str, int] = {}
    domain_fn: Dict[str, int] = {}
    domain_tn: Dict[str, int] = {}

    def _domain_of(prompt_dict: dict) -> str:
        return prompt_dict.get("domain", "general")

    print(f"\n[STATIC BASELINE] Running on {len(Pm)} malicious + {len(Pb)} benign prompts...")

    # ── Malicious prompts ─────────────────────────────────────────────────────
    for i, pm in enumerate(Pm):
        t0 = time.perf_counter()
        text = pm["text"]
        pid = pm.get("id", f"mal-{i:04d}")
        domain = _domain_of(pm)
        normalized = normalize_text(text)

        decision, transformed, trace = executor.apply(ensemble_name, text, {})
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        if decision == "BLOCK":
            llm_output, vflag, vtype = None, False, "BLOCKED_BY_ENSEMBLE"
            jtrace = {}
            # True positive: malicious correctly blocked
            domain_tp[domain] = domain_tp.get(domain, 0) + 1
        else:
            model_input = transformed or text
            llm_output, lat, cost = call_model(model_input, model_name, {}, config)
            latency_ms += lat
            vflag, vtype, jtrace = judge_policy_violation(llm_output, config)
            if vflag is True:
                success_count += 1
                domain_fn[domain] = domain_fn.get(domain, 0) + 1  # missed by ensemble
            else:
                domain_tp[domain] = domain_tp.get(domain, 0) + 1

        logger.log_event(
            experiment_id=exp_id,
            model=model_name,
            ensemble=ensemble_name,
            prompt_id=pid,
            seed_prompt=text,
            normalized_prompt=normalized,
            ensemble_decision=decision,
            transformed_prompt=transformed,
            layer_trace=trace,
            llm_output=llm_output,
            post_judge={"violation_flag": vflag, "violation_type": vtype, **jtrace},
            attack_meta={"attacker": "static"},
            latency_ms=latency_ms,
            cost_estimate=0.0,
            random_seed=seed,
        )

        attack_rows.append({
            "prompt": text[:120],
            "ensemble_decision": decision,
            "transformed_prompt": (transformed or "")[:80],
            "llm_output": (llm_output or "")[:120],
            "judge_flag": vflag,
            "judge_type": vtype,
            "success": vflag is True,
            "fail_layer": _first_block_layer(trace),
            "latency_ms": latency_ms,
            "domain": domain,
        })
        print(f"  [{i+1}/{len(Pm)}] {pid}: {decision} | judge={vflag} ({vtype})")

    # ── Benign prompts ────────────────────────────────────────────────────────
    for i, pb in enumerate(Pb):
        t0 = time.perf_counter()
        text = pb["text"]
        pid = pb.get("id", f"ben-{i:04d}")
        domain = _domain_of(pb)
        normalized = normalize_text(text)

        decision, transformed, trace = executor.apply(ensemble_name, text, {})
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        if decision == "BLOCK":
            benign_refusal_count += 1
            llm_output = None
            vflag, vtype, jtrace = False, "BLOCKED_BENIGN", {}
            # False positive: benign incorrectly blocked
            domain_fp[domain] = domain_fp.get(domain, 0) + 1
        else:
            model_input = transformed or text
            llm_output, lat, cost = call_model(model_input, model_name, {}, config)
            latency_ms += lat
            vflag, vtype, jtrace = judge_policy_violation(llm_output, config)
            domain_tn[domain] = domain_tn.get(domain, 0) + 1

        logger.log_event(
            experiment_id=exp_id,
            model=model_name,
            ensemble=ensemble_name,
            prompt_id=pid,
            seed_prompt=text,
            normalized_prompt=normalized,
            ensemble_decision=decision,
            transformed_prompt=transformed,
            layer_trace=trace,
            llm_output=llm_output,
            post_judge={"violation_flag": vflag, "violation_type": vtype},
            attack_meta={"attacker": "none"},
            latency_ms=latency_ms,
            cost_estimate=0.0,
            random_seed=seed,
        )

        benign_rows.append({
            "prompt": text[:120],
            "ensemble_decision": decision,
            "judge_flag": vflag,
            "judge_type": vtype,
            "blocked": decision == "BLOCK",
            "latency_ms": latency_ms,
            "domain": domain,
        })
        print(f"  [{i+1}/{len(Pb)}] {pid}: {decision}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    asr = round(success_count / len(Pm), 4) if Pm else 0.0
    frr = round(benign_refusal_count / len(Pb), 4) if Pb else 0.0

    # Per-domain precision / recall / F1
    per_domain = {}
    all_domains = set(list(domain_tp) + list(domain_fp) + list(domain_fn) + list(domain_tn))
    for d in all_domains:
        tp = domain_tp.get(d, 0)
        fp = domain_fp.get(d, 0)
        fn = domain_fn.get(d, 0)
        tn = domain_tn.get(d, 0)
        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0
        fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
        fnr = round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0
        per_domain[d] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                         "precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "fnr": fnr}

    # Top 5 failure prompts (malicious that bypassed defense)
    failures = [r for r in attack_rows if r.get("success")]
    top_failures = [(r["prompt"], r.get("fail_layer", "none")) for r in failures[:5]]

    results = {
        "asr": asr,
        "frr": frr,
        "total_malicious": len(Pm),
        "total_benign": len(Pb),
        "successes": success_count,
        "benign_refusals": benign_refusal_count,
        "per_domain": per_domain,
        "top_failures": top_failures,
    }

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    _save_csv(attack_rows, out_dir / "attack_results.csv")
    _save_csv(benign_rows, out_dir / "benign_results.csv")

    print(f"\n[STATIC BASELINE] ASR_static={asr:.3f}  FRR={frr:.3f}")
    print(f"  Malicious blocked or judged safe: {len(Pm) - success_count}/{len(Pm)}")
    print(f"  Benign refused: {benign_refusal_count}/{len(Pb)}")
    return results


def _save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [CSV] Saved {len(rows)} rows -> {path}")
