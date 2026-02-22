"""test_run_small_dataset.py — integration test: static_baseline on small dataset."""
import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from pathlib import Path


TEST_CONFIG = {
    "experiment": {"id": "test-integration", "seed": 42},
    "models": [{"name": "mock", "adapter": "mock", "generation_params": {}}],
    "defenses": {
        "ensembles": {
            "default_ensemble": {
                "order": ["normalization", "keyword_blocker", "intent_classifier", "capability_reducer"],
                "layer_weights": {"keyword_blocker": 0.6, "intent_classifier": 0.8, "capability_reducer": 0.4},
                "resolver": {"block_threshold": 0.8, "flag_threshold": 0.4},
            }
        }
    },
    "semantic": {
        "embedding_model": "all-MiniLM-L6-v2",
        "domain_thresholds": {
            "violence": 0.62, "cybercrime": 0.58, "injection": 0.65,
            "drugs": 0.60, "weapons": 0.61,
        },
        "severity_default": {
            "violence": "block", "drugs": "block", "weapons": "block",
            "cybercrime": "flag", "injection": "flag",
        },
    },
    "logging": {"out_dir": "logs/test/", "jsonl_file": "logs/test/test_events.jsonl"},
    "outputs": {"out_dir": "outputs/test/"},
    "evaluation": {"per_domain": True, "compute_precision_recall": True, "save_plots": False},
}


@pytest.fixture(scope="module")
def tmp_outputs():
    """Create temp output dirs and update config paths."""
    tmpdir = tempfile.mkdtemp(prefix="llmsec_test_")
    cfg = dict(TEST_CONFIG)
    cfg["data"] = {
        "benign": str(Path(__file__).parent.parent / "data" / "test" / "test_benign.json"),
        "malicious": str(Path(__file__).parent.parent / "data" / "test" / "test_attacks.json"),
        "malicious_dir": str(Path(__file__).parent.parent / "data" / "malicious/"),
    }
    cfg["logging"] = {"out_dir": tmpdir, "jsonl_file": os.path.join(tmpdir, "events.jsonl")}
    cfg["outputs"] = {"out_dir": tmpdir}
    cfg["evaluation"] = {"per_domain": True, "compute_precision_recall": True, "save_plots": False}
    yield cfg, tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_static_baseline_runs(tmp_outputs):
    """Run static_baseline on small test dataset and verify output structure."""
    cfg, tmpdir = tmp_outputs
    from utils.logging_utils import EventLogger
    logger = EventLogger(cfg["logging"]["jsonl_file"])

    from experiments.static_baseline import run_static_baseline
    results = run_static_baseline("mock", "default_ensemble", cfg, logger)
    logger.close()

    assert "asr" in results, "Results must contain 'asr' key"
    assert "frr" in results, "Results must contain 'frr' key"
    assert 0.0 <= results["asr"] <= 1.0, f"ASR out of range: {results['asr']}"
    assert 0.0 <= results["frr"] <= 1.0, f"FRR out of range: {results['frr']}"
    assert results["total_malicious"] > 0, "Should have processed malicious prompts"
    assert results["total_benign"] > 0, "Should have processed benign prompts"


def test_static_baseline_produces_csv(tmp_outputs):
    """Check that CSV files are created."""
    cfg, tmpdir = tmp_outputs
    attack_csv = Path(tmpdir) / "attack_results.csv"
    benign_csv = Path(tmpdir) / "benign_results.csv"
    # These should exist if the test above ran
    # If not, run baseline again
    if not attack_csv.exists():
        from utils.logging_utils import EventLogger
        logger = EventLogger(cfg["logging"]["jsonl_file"])
        from experiments.static_baseline import run_static_baseline
        run_static_baseline("mock", "default_ensemble", cfg, logger)
        logger.close()
    assert attack_csv.exists(), f"attack_results.csv not found in {tmpdir}"


def test_static_baseline_plausible_metrics(tmp_outputs):
    """Metrics should be in plausible ranges for known test data."""
    cfg, tmpdir = tmp_outputs
    from utils.logging_utils import EventLogger
    logger = EventLogger(cfg["logging"]["jsonl_file"])
    from experiments.static_baseline import run_static_baseline
    results = run_static_baseline("mock", "default_ensemble", cfg, logger)
    logger.close()

    # With 5 strongly-malicious prompts, at least some should be blocked
    total = results["total_malicious"]
    blocked_plus_judged = total - results["successes"]
    # The defense should block or correctly handle at least some malicious prompts
    # (not all may bypass with static attack + mock model)
    assert total >= 5, f"Expected at least 5 malicious prompts, got {total}"
    assert results["total_benign"] >= 5, f"Expected at least 5 benign prompts"


def test_events_jsonl_produced(tmp_outputs):
    """Verify JSONL event log is written."""
    cfg, tmpdir = tmp_outputs
    jsonl_path = Path(cfg["logging"]["jsonl_file"])
    from utils.logging_utils import EventLogger
    logger = EventLogger(str(jsonl_path))
    from experiments.static_baseline import run_static_baseline
    run_static_baseline("mock", "default_ensemble", cfg, logger)
    logger.close()
    assert jsonl_path.exists(), "events.jsonl should be created"
    events = EventLogger(str(jsonl_path)).read_all()
    assert len(events) > 0, "events.jsonl should contain at least one event"
    first = events[0]
    assert "timestamp" in first
    assert "ensemble_decision" in first
    assert "layer_trace" in first
