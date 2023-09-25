"""Cifar10 experiments."""

from collections.abc import Sequence
import os
import functools
from absl import app
from absl import flags
import jax
import jax.numpy as jnp
from ml_collections import config_flags

from sddm.common import train_eval
from sddm.common import utils
from sddm.image.cifar10 import cifar10_utils
from sddm.image.cifar10 import data_loader

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
_CONFIG = config_flags.DEFINE_config_file('config', lock_config=False)
flags.DEFINE_string('data_root', None, 'data folder')
flags.DEFINE_integer('seed', 1023, 'random seed')
FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = _CONFIG.value
  cifar10_utils.postprocess_config(config)
  writer = train_eval.setup_logging(config)
  global_key = jax.random.PRNGKey(FLAGS.seed)
  global_key, model_key = jax.random.split(global_key, 2)
  if config.model_type.startswith('vq_'):
    config.model_type = config.model_type[len('vq_'):]
    if config.model_type == 'd3pm':
      model = cifar10_utils.D3pmVqCifar10(config)
    else:
      model = cifar10_utils.CtVqCifar10(config)
  elif config.model_type == 'hollow' or config.model_type == 'ordinal':
    model = cifar10_utils.CtCifar10(config)
  else:
    raise ValueError('Unknown model type %s' % config.model_type)
  state = model.init_state(model_key)
  sample_fn = jax.pmap(model.sample_loop, axis_name='shard')
  fn_plot = functools.partial(model.plot, sample_fn=sample_fn, writer=writer)
  if config.phase == 'train':
    train_ds = utils.numpy_iter(data_loader.get_dataloader(config, 'train'))
    train_eval.train_loop(
        config, writer=writer, global_key=global_key,
        train_step_fn=model.step_fn, train_ds=train_ds, state=state,
        fn_plot_data=functools.partial(model.plot_data, writer=writer),
        fn_eval=fn_plot, fn_data_preprocess=model.encode_batch,
    )
  else:
    assert os.path.exists(config.model_init_folder)
    if config.phase == 'eval_score':
      evaluator = cifar10_utils.Cifar10FidIs(
          config, model.sample_loop, writer=writer)
      fn_eval = evaluator.eval_model
    elif config.phase == 'plot':
      fn_eval = fn_plot
    else:
      raise ValueError('Unknown phase %s' % config.phase)
    train_eval.eval_latest_model(
        config.model_init_folder, writer=writer, global_key=global_key,
        state=state, fn_eval=fn_eval)
  # Wait until all processes are done before exiting.
  jax.pmap(jax.random.PRNGKey)(jnp.arange(
      jax.local_device_count())).block_until_ready()


if __name__ == '__main__':
  app.run(main)
