"""Synthetic experiments on gray-code."""

from collections.abc import Sequence
import functools
import os
from absl import app
from absl import flags
from flax import jax_utils
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import yaml

from sddm.common import train_eval
from sddm.common import utils
from sddm.model import continuous_time_diffusion
from sddm.model import discrete_time_diffusion
from sddm.synthetic import common as synthetic_common
from sddm.synthetic.data import data_loader
from sddm.synthetic.data import utils as data_utils

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_CONFIG = config_flags.DEFINE_config_file('config', lock_config=False)
flags.DEFINE_string('data_root', None, 'data folder')
flags.DEFINE_integer('seed', 1023, 'random seed')
FLAGS = flags.FLAGS


class CategoricalSyntheticHelper(object):
  def __init__(self, config):
    self.config = config

  def plot(self, xbase, output_file=None):
    fn_xbase2float = functools.partial(
        data_utils.ourbase2float,
        discrete_dim=self.config.discrete_dim, f_scale=self.config.f_scale ,int_scale=self.config.int_scale, vocab_size=self.config.vocab_size)
    return synthetic_common.plot(xbase, fn_xbase2float, output_file)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = _CONFIG.value
  data_folder = os.path.join(FLAGS.data_root, config.data_folder)
  with open(os.path.join(data_folder, 'config.yaml'), 'r') as f:
    data_config = yaml.unsafe_load(f)
  config.update(data_config)
  config.data_folder = data_folder
  global_key = jax.random.PRNGKey(FLAGS.seed)

  train_ds = utils.numpy_iter(data_loader.get_dataloader(config))
  global_key, model_key = jax.random.split(global_key, 2)
  if config.model_type == 'd3pm':
    model = discrete_time_diffusion.D3PM(config)
  else:
    model = continuous_time_diffusion.CategoricalDiffusionModel(config)
  model_helper = CategoricalSyntheticHelper(config)
  state = model.init_state(model_key)
  writer = train_eval.setup_logging(config)

  sample_fn = jax.pmap(model.sample_loop, axis_name='shard')
  fn_metric = jax.jit(utils.binary_exp_hamming_mmd)

  def eval_mmd(state, rng):
    """Eval mmd."""
    assert jax.process_count() == 1
    avg_mmd = 0.0
    result = []
    for i in range(config.eval_rounds):
      gt_data = []
      for _ in range(config.plot_samples // config.batch_size):
        gt_data.append(next(train_ds))
      if len(gt_data) > 1:
        gt_data = jnp.concatenate(gt_data, axis=0)
      else:
        gt_data = gt_data[0]
      gt_data = jnp.reshape(gt_data, (-1, config.discrete_dim))
      rng = jax.random.fold_in(rng, i)
      step_rng_keys = utils.shard_prng_key(rng)
      x0 = sample_fn(state, step_rng_keys)
      x0 = utils.all_gather(x0)
      x0 = jnp.reshape(jax_utils.unreplicate(x0), gt_data.shape)
      result.append(jax.device_get(x0))
      mmd = fn_metric(x0, gt_data)
      avg_mmd += mmd
    mmd = avg_mmd / config.eval_rounds

    return mmd, result

  train_eval.train_loop(
      config, writer=writer, global_key=global_key, train_step_fn=model.step_fn,
      train_ds=train_ds, state=state,
      fn_plot_data=functools.partial(
          synthetic_common.fn_plot_data, config=config,
          model=model_helper, writer=writer),
      fn_eval=functools.partial(
          synthetic_common.fn_eval_with_mmd, eval_mmd=eval_mmd,
          config=config, model=model_helper, writer=writer))


if __name__ == '__main__':
  app.run(main)
