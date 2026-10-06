"""
reproducibility.py
Handles deterministic seed initialization across Python, NumPy, PyTorch,
and records environment configuration to reports/reproducibility.json.
"""

import os
import random
import sys
import json
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds for random, numpy, and torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
        
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def record_reproducibility_report(extra_info: Dict[str, Any] = None) -> Path:
    """Saves system environment, packages, and seeds to reports/reproducibility.json."""
    report_path = PROJECT_ROOT / "reports" / "reproducibility.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    packages = {}
    for pkg in ["torch", "torchvision", "numpy", "scipy", "pandas", "sklearn", "cv2", "streamlit"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[pkg] = "not installed"
            
    info = {
        "random_seed": 42,
        "python_version": sys.version,
        "platform": sys.platform,
        "packages": packages,
        "cpu_count": os.cpu_count(),
        "extra_info": extra_info or {}
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
        
    return report_path


if __name__ == "__main__":
    set_seed(42)
    path = record_reproducibility_report({"status": "initialized"})
    print(f"Reproducibility report saved to {path}")
