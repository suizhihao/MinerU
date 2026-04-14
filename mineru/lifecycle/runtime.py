from pathlib import Path
from typing import Any

import torch
from loguru import logger

from .config import LifecycleConfig


def _unwrap_state_dict(loaded: Any):
    if isinstance(loaded, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            maybe = loaded.get(key)
            if isinstance(maybe, dict):
                return maybe
    return loaded


def _resolve_torch_target(model_obj: Any):
    if hasattr(model_obj, "load_state_dict"):
        return model_obj
    if hasattr(model_obj, "model") and hasattr(model_obj.model, "load_state_dict"):
        return model_obj.model
    if hasattr(model_obj, "net") and hasattr(model_obj.net, "load_state_dict"):
        return model_obj.net
    return None


def apply_weight_override(
    role: str, model_obj: Any, config: LifecycleConfig, device: Any = "cpu"
) -> bool:
    if not config.enabled:
        return False
    path = config.resolved_weight_overrides().get(role)
    if not path:
        return False
    ckpt_path = Path(path).expanduser().resolve()
    if not ckpt_path.exists():
        logger.warning("Lifecycle weight override missing (role={}): {}", role, ckpt_path)
        return False

    target = _resolve_torch_target(model_obj)
    if target is None:
        logger.warning(
            "Lifecycle weight override skipped (role={}): no load_state_dict target",
            role,
        )
        return False

    try:
        loaded = torch.load(str(ckpt_path), map_location=device)
        state_dict = _unwrap_state_dict(loaded)
        missing, unexpected = target.load_state_dict(state_dict, strict=False)
        logger.info(
            "Lifecycle weight override loaded for role={} from {} (missing={}, unexpected={})",
            role,
            ckpt_path,
            len(missing),
            len(unexpected),
        )
        return True
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Lifecycle weight override failed for role={} path={} err={}",
            role,
            ckpt_path,
            exc,
        )
        return False
