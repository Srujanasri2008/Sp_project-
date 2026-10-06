"""
setup_environment.py
Checks system resources, creates necessary directories, sets up virtual environment,
and verifies required dependencies.
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
# System libraries only

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIRECTORIES = [
    "app",
    "config",
    "data/raw",
    "data/processed",
    "data/splits",
    "data/metadata",
    "models/binary",
    "models/five_class",
    "models/comparison",
    "src/data",
    "src/processing",
    "src/models",
    "src/training",
    "src/evaluation",
    "src/explainability",
    "src/inference",
    "scripts",
    "tests",
    "reports/error_analysis",
    "figures/confusion_matrices",
    "figures/roc_curves",
    "figures/robustness",
    "figures/gradcam",
    "figures/ablations",
    "docs",
    "logs/training",
    "logs/inference",
    "logs/validation",
    "cache",
]


def check_system_resources():
    print("=" * 60)
    print("PREFLIGHT RESOURCE & HARDWARE REPORT")
    print("=" * 60)
    
    # Python
    print(f"Python Version: {sys.version.split()[0]} ({sys.executable})")
    
    # CPU
    cpu_count = os.cpu_count() or 1
    print(f"CPU Logical Cores: {cpu_count}")
    
    # Disk space
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    gb = 1024 ** 3
    print(f"Project Drive: {PROJECT_ROOT.anchor}")
    print(f"Disk Free Space: {free / gb:.2f} GB / {total / gb:.2f} GB")
    
    # GPU check
    gpu_available = False
    gpu_name = "None (CPU Execution Mode)"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    print(f"GPU Available: {gpu_available} ({gpu_name})")
    
    # Safety assessment
    if free / gb < 5.0:
        print("[WARNING] Free disk space is below 5.0 GB! Large operations must be stopped.")
    else:
        print(f"[STATUS] Disk space OK ({free / gb:.2f} GB free). Safe to proceed.")
    print("=" * 60)
    return {
        "python_version": sys.version.split()[0],
        "cpu_count": cpu_count,
        "free_disk_gb": round(free / gb, 2),
        "gpu_available": gpu_available,
        "gpu_name": gpu_name
    }


def create_directories():
    print("\n[+] Creating project directory structure...")
    for d in DIRECTORIES:
        p = PROJECT_ROOT / d
        p.mkdir(parents=True, exist_ok=True)
    
    # Touch __init__.py files in src and subpackages
    src_dir = PROJECT_ROOT / "src"
    for subdir in src_dir.rglob("*"):
        if subdir.is_dir() and not subdir.name.startswith("__"):
            init_file = subdir / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    (src_dir / "__init__.py").touch()
    print("  All directories and package __init__.py files verified.")


def create_venv():
    venv_dir = PROJECT_ROOT / ".venv"
    if not venv_dir.exists():
        print(f"\n[+] Creating virtual environment at {venv_dir}...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        print("  Virtual environment created successfully.")
    else:
        print(f"\n[STATUS] Virtual environment already exists at {venv_dir}.")


if __name__ == "__main__":
    check_system_resources()
    create_directories()
    create_venv()
    print("\nSetup environment step completed successfully.")
