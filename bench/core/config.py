from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
