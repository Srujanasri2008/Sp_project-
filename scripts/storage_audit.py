"""
storage_audit.py
Audits disk space usage of project directories, models, datasets, caches, and reports.
"""

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError):
        pass
    return total


def format_bytes(b: int) -> str:
    mb = b / (1024 ** 2)
    if mb > 1024:
        return f"{mb / 1024:.2f} GB ({b:,} bytes)"
    return f"{mb:.2f} MB ({b:,} bytes)"


def run_storage_audit():
    print("=" * 65)
    print("STORAGE AND RESOURCE SAFETY AUDIT")
    print("=" * 65)
    
    total_disk, used_disk, free_disk = shutil.disk_usage(PROJECT_ROOT)
    gb = 1024 ** 3
    print(f"Project Location : {PROJECT_ROOT}")
    print(f"Drive Root       : {PROJECT_ROOT.anchor}")
    print(f"Total Disk Space : {total_disk / gb:.2f} GB")
    print(f"Used Disk Space  : {used_disk / gb:.2f} GB")
    print(f"Free Disk Space  : {free_disk / gb:.2f} GB")
    print("-" * 65)
    
    target_dirs = {
        "Project Total": PROJECT_ROOT,
        "Dataset (data/)": PROJECT_ROOT / "data",
        "Raw Data (data/raw/)": PROJECT_ROOT / "data" / "raw",
        "Processed (data/processed/)": PROJECT_ROOT / "data" / "processed",
        "Models (models/)": PROJECT_ROOT / "models",
        "Cache (cache/)": PROJECT_ROOT / "cache",
        "Reports (reports/)": PROJECT_ROOT / "reports",
        "Figures (figures/)": PROJECT_ROOT / "figures",
        "Logs (logs/)": PROJECT_ROOT / "logs",
        "Virtual Env (.venv/)": PROJECT_ROOT / ".venv",
    }
    
    for label, dpath in target_dirs.items():
        size = get_dir_size_bytes(dpath)
        print(f"  {label:<30}: {format_bytes(size)}")
        
    print("-" * 65)
    if (free_disk / gb) < 5.0:
        print("[ALERT] Free disk space is CRITICALLY LOW (< 5.0 GB). STOP further downloads/caches!")
        status = "CRITICAL"
    elif (free_disk / gb) < 10.0:
        print("[WARNING] Free disk space is low (< 10.0 GB). Exercise conservative caching.")
        status = "WARNING"
    else:
        print("[STATUS] Disk headroom is healthy. Safe for conservative execution.")
        status = "HEALTHY"
    print("=" * 65)
    return {
        "status": status,
        "free_disk_gb": round(free_disk / gb, 2),
        "total_disk_gb": round(total_disk / gb, 2)
    }


if __name__ == "__main__":
    run_storage_audit()
