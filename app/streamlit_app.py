"""Compatibility entry point for the documented Streamlit command."""

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_FILE = PROJECT_ROOT / "app" / "app.py"
APP_SPEC = importlib.util.spec_from_file_location("dr_streamlit_ui", APP_FILE)
if APP_SPEC is None or APP_SPEC.loader is None:
    raise ImportError(f"Unable to load Streamlit application from {APP_FILE}")
APP_MODULE = importlib.util.module_from_spec(APP_SPEC)
APP_SPEC.loader.exec_module(APP_MODULE)
main = APP_MODULE.main


if __name__ == "__main__":
    main()