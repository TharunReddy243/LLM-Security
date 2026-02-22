"""First-failure layer analysis — histogram of which layer first blocked a prompt."""
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List


def compute_first_failure_stats(events: List[Dict]) -> Dict[str, int]:
    """Count first-block layer occurrences from logged events."""
    counts: Counter = Counter()
    for ev in events:
        trace = ev.get("layer_trace", [])
        for t in trace:
            if t.get("action") == "BLOCK":
                counts[t["layer"]] += 1
                break
    return dict(counts)


def save_first_failure_csv(counts: Dict[str, int], out_dir: str) -> str:
    path = Path(out_dir) / "analysis" / "first_failure_stats.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "first_block_count"])
        for layer, count in sorted(counts.items(), key=lambda x: -x[1]):
            writer.writerow([layer, count])
    return str(path)
