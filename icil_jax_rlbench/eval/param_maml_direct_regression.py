from __future__ import annotations

from typing import Sequence

from absl import app
from ml_collections.config_flags import config_flags

from icil_jax_rlbench.eval.online_common import evaluate


_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='icil_jax_rlbench/configs/eval_online_param_maml.py',
    help_string='Path to online param-MAML/FOMAML evaluation config.',
)


def main(argv: Sequence[str]) -> None:
    del argv
    evaluate(_CONFIG.value, adaptation='param_maml')


if __name__ == '__main__':
    app.run(main)

