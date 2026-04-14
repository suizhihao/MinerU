import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


def _as_bool(raw: Optional[str], default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_with_legacy(
    lifecycle_key: str,
    legacy_governance_key: Optional[str] = None,
) -> Optional[str]:
    primary = os.getenv(lifecycle_key)
    if primary is not None and str(primary).strip() != "":
        return primary
    if legacy_governance_key:
        legacy = os.getenv(legacy_governance_key)
        if legacy is not None and str(legacy).strip() != "":
            logger.warning(
                "{} is deprecated; use {}",
                legacy_governance_key,
                lifecycle_key,
            )
            return legacy
    return None


@dataclass(frozen=True)
class LifecycleConfig:
    enabled: bool
    layout_checkpoint: Optional[str]
    ocr_checkpoint: Optional[str]
    weight_overrides_json: Optional[str]

    def resolved_weight_overrides(self) -> Dict[str, str]:
        merged: Dict[str, str] = {}
        if self.weight_overrides_json:
            json_path = Path(self.weight_overrides_json)
            if json_path.exists():
                try:
                    loaded = json.loads(json_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        merged.update({str(k): str(v) for k, v in loaded.items()})
                    else:
                        logger.warning(
                            "weight overrides JSON should map role->path, got: {}",
                            type(loaded).__name__,
                        )
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to load weight override json {}: {}", json_path, exc)
            else:
                logger.warning("Weight override json not found: {}", json_path)

        if self.layout_checkpoint:
            merged["layout"] = self.layout_checkpoint
        if self.ocr_checkpoint:
            merged["ocr"] = self.ocr_checkpoint
        return merged


def load_lifecycle_config() -> LifecycleConfig:
    enable_raw = _env_with_legacy(
        "MINERU_LIFECYCLE_ENABLE",
        "MINERU_GOVERNANCE_ENABLE",
    )
    layout = _env_with_legacy(
        "MINERU_LIFECYCLE_LAYOUT_CHECKPOINT",
        "MINERU_GOVERNANCE_LAYOUT_CHECKPOINT",
    )
    ocr = _env_with_legacy(
        "MINERU_LIFECYCLE_OCR_CHECKPOINT",
        "MINERU_GOVERNANCE_OCR_CHECKPOINT",
    )
    wjson = _env_with_legacy(
        "MINERU_LIFECYCLE_WEIGHT_OVERRIDES_JSON",
        "MINERU_GOVERNANCE_WEIGHT_OVERRIDES_JSON",
    )
    return LifecycleConfig(
        enabled=_as_bool(enable_raw, default=False),
        layout_checkpoint=layout,
        ocr_checkpoint=ocr,
        weight_overrides_json=wjson,
    )
