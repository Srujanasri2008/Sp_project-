"""Run the project tests and preserve their output under logs/validation."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    validation_dir = PROJECT_ROOT / "logs" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "-q"]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    log_path = validation_dir / "pytest_latest.txt"
    log_path.write_text(
        f"Command: {' '.join(command)}\n"
        f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}\n"
        f"Exit code: {result.returncode}\n\n{output}",
        encoding="utf-8",
    )
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "exit_code": result.returncode,
        "log_file": str(log_path.relative_to(PROJECT_ROOT)),
    }
    (validation_dir / "pytest_latest.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output, end="")
    print(f"Validation log saved to: {log_path.relative_to(PROJECT_ROOT)}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()