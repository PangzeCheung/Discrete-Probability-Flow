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


def test_variance(argv: Sequence[str]) -> None:
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
  state = model.init_state(model_key)
  writer = train_eval.setup_logging(config)

  sample_fn = jax.pmap(model.sample_loop_variance, axis_name='shard')
  fn_metric = jax.jit(utils.binary_exp_hamming_mmd)

  def eval(state, rng):
    """Eval mmd."""
    assert jax.process_count() == 1

    rng_temp, prior_rng = jax.random.split(rng)
    num_samples = config.plot_samples // jax.device_count()
    x_start = model.sample_from_prior(prior_rng, num_samples)

    x_start = jax_utils.replicate(x_start)

    result = []

    for i in range(10):
      rng = jax.random.fold_in(rng, i)
      step_rng_keys = utils.shard_prng_key(rng)
      x0 = sample_fn(state, step_rng_keys, x_start=x_start)
      x0 = utils.all_gather(x0)
      x0 = jnp.reshape(jax_utils.unreplicate(x0), (-1, config.discrete_dim))
      result.append(jnp.expand_dims(jax.device_get(x0), axis=-1))

    result = jnp.concatenate(result, axis=-1)

    return result

  train_eval.eval_variance_latest_model(config.save_root, writer, global_key, state, fn_eval=functools.partial(
          synthetic_common.fn_test_variance, eval=eval,
          config=config, model=None, writer=writer), prefix='bestckpt_')



if __name__ == '__main__':
  app.run(test_variance)
