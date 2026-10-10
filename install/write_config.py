"""Create/refresh <POLONY_HOME>/config.json without overwriting values the user already has."""
import json
import os
import pathlib

p = pathlib.Path(os.environ["POLONY_HOME"]) / "config.json"
try:
    cfg = json.loads(p.read_text())
except Exception:
    cfg = {}
cfg.setdefault("data_dir", os.environ["POLONY_CFG_DATA"])
cfg.setdefault("port", 8765)
p.write_text(json.dumps(cfg, indent=2))
