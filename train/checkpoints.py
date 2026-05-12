from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import jax


def _to_host(x: Any) -> Any:
    return jax.device_get(x)


def unreplicate_state(state: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: x[0] if hasattr(x, 'shape') and len(x.shape) > 0 else x, state)


def save_checkpoint(path: str | Path, *, state: Any, step: int, config: Any = None, extra: Optional[Dict[str, Any]] = None, replicated: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    st = unreplicate_state(state) if replicated else state
    cfg = config.to_dict() if hasattr(config, 'to_dict') else config
    payload = {
        'step': int(step),
        'params': _to_host(st.params),
        'opt_state': _to_host(st.opt_state),
        'rng': _to_host(st.rng),
        'config': cfg,
        'extra': extra or {},
    }
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    with Path(path).open('rb') as f:
        return pickle.load(f)
