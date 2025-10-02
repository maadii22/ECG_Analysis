"""
Project-wide configuration. Exposes DATA_ROOT, MODELS_DIR and RESULTS_DIR.
Paths can be overridden with the environment variable ECG_DATA_ROOT.
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent

# Allow user to override dataset root via env var
DATA_ROOT = Path(os.environ.get("ECG_DATA_ROOT", PROJECT_ROOT / "datasets")).resolve()
MODELS_DIR = DATA_ROOT / "models"
RESULTS_DIR = DATA_ROOT / "results"

# Ensure directories exist
for d in (DATA_ROOT, MODELS_DIR, RESULTS_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best-effort; some environments may restrict permissions
        pass

__all__ = ["PROJECT_ROOT", "DATA_ROOT", "MODELS_DIR", "RESULTS_DIR"]
