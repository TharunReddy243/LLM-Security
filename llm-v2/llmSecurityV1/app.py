"""
GuardAI — FastAPI Backend
=========================
Place this file in your project root (same level as run.py).

Install deps (once):
    pip install fastapi uvicorn

Run:
    uvicorn app:app --reload --port 8000

The frontend (index.html) can be opened directly in browser,
or served via:  python -m http.server 3000  (in same folder)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys, os, time

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import load_config
from defenses.ensemble import EnsembleExecutor
import model.model_api as _model_api
from model.model_api import call_model
from judge.policy_judge import judge_policy_violation

# Enable real OpenRouter responses for the web frontend
_model_api.WEB_MODE = True

# ── Bootstrap once at startup ─────────────────────────────────────────────────
# Set GROQ_API_KEY so model_api can call Llama 3 for real responses on benign prompts.
# Groq is FREE — sign up at console.groq.com, create an API key, then:
#   Windows CMD:         set GROQ_API_KEY=gsk_...
#   Windows PowerShell:  $env:GROQ_API_KEY="gsk_..."
#   Mac/Linux:           export GROQ_API_KEY=gsk_...
# Or hardcode it here for testing (not recommended for production):
# os.environ.setdefault("GROQ_API_KEY", "gsk_YOUR_KEY_HERE")

config   = load_config("config.yaml")
executor = EnsembleExecutor(config)
ENSEMBLE = "default_ensemble"
MODEL    = "mock"

# ── Domain metadata ───────────────────────────────────────────────────────────
DOMAIN_LABELS = {
    "cybercrime": "Cybercrime & Hacking",
    "weapons":    "Weapons & Explosives",
    "drugs":      "Drug Manufacturing",
    "violence":   "Violence & Harm",
    "fraud":      "Financial Fraud",
    "evasion":    "Law Evasion",
    "injection":  "Prompt Injection",
    "none":       "Policy Violation",
    "general":    "Policy Violation",
}
THREAT_LEVELS = {
    "cybercrime": "HIGH",
    "weapons":    "CRITICAL",
    "drugs":      "HIGH",
    "violence":   "CRITICAL",
    "fraud":      "MEDIUM",
    "evasion":    "MEDIUM",
    "injection":  "HIGH",
    "none":       "MEDIUM",
    "general":    "LOW",
}
LAYER_LABELS = {
    "normalization":        "L1 · Normalization",
    "keyword_blocker":      "L2 · Keyword Blocker",
    "intent_classifier":    "L3 · Intent Classifier",
    "capability_reducer":   "L4 · Capability Reducer",
    "intent_masking_detector": "L5 · Masking Detector",
    "resolver":             "Resolver",
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="GuardAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


class ChatRequest(BaseModel):
    prompt: str


def _get_domain(trace: list) -> str:
    for t in trace:
        d = t.get("meta", {}).get("domain", "")
        if d and d != "none":
            return d
    return "none"


def _first_block_layer(trace: list) -> str:
    for t in trace:
        if t.get("action") == "BLOCK":
            return t.get("layer", "unknown")
    return "resolver"


def _format_trace(trace: list) -> list:
    out = []
    for t in trace:
        out.append({
            "layer":      LAYER_LABELS.get(t.get("layer", ""), t.get("layer", "?")),
            "action":     t.get("action", "ALLOW"),
            "reason":     t.get("meta", {}).get("reason", ""),
            "latency_ms": t.get("latency_ms", 0),
        })
    return out


@app.post("/api/chat")
def chat(req: ChatRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(400, "Empty prompt")

    t0 = time.perf_counter()
    try:
        decision, transformed, trace = executor.apply(ENSEMBLE, prompt, {})
        total_ms = round((time.perf_counter() - t0) * 1000, 1)

        if decision == "BLOCK":
            domain      = _get_domain(trace)
            block_layer = _first_block_layer(trace)
            return {
                "blocked":      True,
                "category":     DOMAIN_LABELS.get(domain, "Policy Violation"),
                "threat_level": THREAT_LEVELS.get(domain, "HIGH"),
                "domain":       domain,
                "block_layer":  LAYER_LABELS.get(block_layer, block_layer),
                "trace":        _format_trace(trace),
                "latency_ms":   total_ms,
            }

        model_input            = transformed or prompt
        llm_output, lat, _     = call_model(model_input, MODEL, {}, config)
        vflag, vtype, _jtrace  = judge_policy_violation(llm_output, config)
        total_ms += lat

        # If the model output is flagged as harmful, block it — don't show it to the user.
        # This handles cases where a malicious prompt slipped past the defense layers
        # and the model produced a harmful response. The judge catches it here.
        if vflag is True:
            domain      = _get_domain(trace)
            return {
                "blocked":      True,
                "category":     DOMAIN_LABELS.get(domain, "Policy Violation"),
                "threat_level": THREAT_LEVELS.get(domain, "HIGH"),
                "domain":       domain,
                "block_layer":  "Output Judge",
                "trace":        _format_trace(trace),
                "latency_ms":   round(total_ms, 1),
            }

        return {
            "blocked":    False,
            "response":   llm_output,
            "flagged":    False,
            "flag_type":  vtype,
            "trace":      _format_trace(trace),
            "latency_ms": round(total_ms, 1),
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL, "ensemble": ENSEMBLE}