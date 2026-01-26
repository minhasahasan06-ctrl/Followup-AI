"""
GCP Configuration package.
Re-exports settings from app/config.py for backward compatibility.
"""

import sys
import importlib.util
from pathlib import Path

config_file_path = Path(__file__).parent.parent / "config.py"
spec = importlib.util.spec_from_file_location("config_module", str(config_file_path))
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

settings = config_module.settings
check_openai_baa_compliance = config_module.check_openai_baa_compliance
Settings = config_module.Settings
get_openai_client = config_module.get_openai_client
