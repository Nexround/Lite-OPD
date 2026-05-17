from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        record = dict(record)
        if "time" in record and isinstance(record["time"], (int, float)):
            record["time"] = datetime.fromtimestamp(record["time"]).strftime("%Y-%m-%d-%H%M%S")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
