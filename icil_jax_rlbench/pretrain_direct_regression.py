from absl import app
from ml_collections import config_flags

from icil_jax_rlbench.train.runner import train

_CONFIG = config_flags.DEFINE_config_file('config', None, 'Training config.', lock_config=False)


def main(argv):
    del argv
    train('pretrain', _CONFIG.value)


if __name__ == '__main__':
    app.run(main)
