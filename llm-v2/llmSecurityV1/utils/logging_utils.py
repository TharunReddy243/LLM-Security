"""Structured JSONL event logger."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class EventLogger:
    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        if "timestamp" not in record:
            record["timestamp"] = _utcnow()
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def log_event(
        self,
        experiment_id: str,
        model: str,
        ensemble: str,
        prompt_id: str,
        seed_prompt: str,
        normalized_prompt: str,
        ensemble_decision: str,
        transformed_prompt: Optional[str],
        layer_trace: list,
        llm_output: Optional[str],
        post_judge: Optional[Dict],
        attack_meta: Optional[Dict],
        latency_ms: float,
        cost_estimate: float,
        random_seed: int,
    ) -> None:
        self.log({
            "timestamp": _utcnow(),
            "experiment_id": experiment_id,
            "model": model,
            "ensemble": ensemble,
            "prompt_id": prompt_id,
            "seed_prompt": seed_prompt,
            "normalized_prompt": normalized_prompt,
            "ensemble_decision": ensemble_decision,
            "transformed_prompt": transformed_prompt,
            "layer_trace": layer_trace,
            "llm_output": llm_output,
            "post_judge": post_judge,
            "attack": attack_meta,
            "latency_ms": latency_ms,
            "cost_estimate": cost_estimate,
            "random_seed": random_seed,
        })

    def close(self):
        self._fh.close()

    def read_all(self) -> list:
        if not self.path.exists():
            return []
        records = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records
