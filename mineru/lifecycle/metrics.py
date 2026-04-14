import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


def _metrics_enabled() -> bool:
    raw = os.getenv("MINERU_LIFECYCLE_METRICS_ENABLE", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _metrics_path() -> Optional[Path]:
    raw = os.getenv("MINERU_LIFECYCLE_METRICS_PATH")
    if not raw or not str(raw).strip():
        return None
    return Path(raw).expanduser().resolve()


def record_pipeline_batch(
    *,
    batch_index: int,
    page_count: int,
    infer_seconds: float,
    doc_slices: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not _metrics_enabled():
        return
    out = _metrics_path()
    if out is None:
        logger.warning(
            "MINERU_LIFECYCLE_METRICS_ENABLE is set but MINERU_LIFECYCLE_METRICS_PATH is empty; skipping metrics write"
        )
        return
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "batch_index": batch_index,
        "page_count": page_count,
        "infer_seconds": round(infer_seconds, 4),
        "doc_slices": doc_slices,
    }
    run_id = os.getenv("MINERU_LIFECYCLE_METRICS_RUN_ID", "").strip()
    if run_id:
        row["run_id"] = run_id
    if extra:
        row.update(extra)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
