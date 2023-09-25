"""Conditional generation of piano data."""

from collections.abc import Sequence
import functools
import os
from absl import app
from absl import flags
from flax import jax_utils
import jax
import jax.numpy as jnp
from ml_collections import config_flags

from sddm.common import train_eval
from sddm.common import utils
from sddm.model import continuous_time_diffusion
from sddm.sequence.piano import data_loader


_CONFIG = config_flags.DEFINE_config_file('config', lock_config=False)
flags.DEFINE_string('data_root', None, 'data folder')
flags.DEFINE_integer('seed', 1023, 'random seed')
FLAGS = flags.FLAGS


class PianoModel(continuous_time_diffusion.PrefixCondCategoricalDiffusionModel):
  """Model Categorical Music."""

  @functools.partial(jax.pmap, static_broadcasted_argnums=(0,))
  def eval_step(self, state, rng, batch):
    mask, batch = jnp.split(batch, [1], axis=1)
    conditioner = batch[:, :self.config.conditional_dim]
    x0 = self.sample_loop(state, rng=rng, num_samples=batch.shape[0],
                          conditioner=conditioner)
    x0_onehot = jax.nn.one_hot(x0, self.config.vocab_size)
    generate_note_cnt = jnp.sum(x0_onehot, axis=1)
    gt_note_cnt = jnp.sum(jax.nn.one_hot(batch, self.config.vocab_size), axis=1)
    mask = mask.astype(jnp.float32)
    sample_cnt = jnp.sum(mask)
    generate_note_cnt = generate_note_cnt * mask
    gt_note_cnt = gt_note_cnt * mask
    outlier = generate_note_cnt * (gt_note_cnt < 0.1)
    frac = jnp.sum(outlier, axis=-1) / batch.shape[1]
    frac_sqr = frac ** 2
    dist_gen = generate_note_cnt / (jnp.sum(
        generate_note_cnt, axis=1, keepdims=True) + 1e-20)
    dist_gt = gt_note_cnt / (jnp.sum(
        gt_note_cnt, axis=1, keepdims=True) + 1e-20)
    distance = 1.0 / jnp.sqrt(2) * jnp.sqrt(jnp.sum(
        (jnp.sqrt(dist_gen) - jnp.sqrt(dist_gt)) ** 2, axis=1))
    aux = {
        'num_samples': sample_cnt,
        'outlier_frac': frac,
        'outlier_frac_sqr': frac_sqr,
        'hellinger_dist': distance,
        'hellinger_dist_sqr': distance ** 2,
    }
    return aux

  def eval_loop(self, step, state, rng, test_tf_ds, writer):
    local_aux = None
    test_ds = utils.numpy_iter(test_tf_ds)
    for i, batch in enumerate(test_ds):
      rng = jax.random.fold_in(rng, i)
      for _ in range(self.config.eval_rounds):
        rng, _ = jax.random.split(rng)
        step_rng_keys = utils.shard_prng_key(rng)
        aux = self.eval_step(state, step_rng_keys, batch)
        if local_aux is None:
          local_aux = aux
        else:
          local_aux = jax.tree_map(jnp.add, aux, local_aux)
    local_aux = jax.tree_map(lambda x: jnp.expand_dims(x, axis=1), local_aux)
    aux = utils.all_gather(local_aux)
    aux = jax.device_get(jax_utils.unreplicate(aux))
    num_samples = jnp.sum(aux['num_samples'])
    metrics = {'num_samples': num_samples}
    for key in aux:
      if key != 'num_samples' and 'sqr' not in key:
        avg = jnp.sum(aux[key]) / num_samples
        avg_sqr = jnp.sum(aux[key + '_sqr']) / num_samples
        metrics['eval/' + key] = avg
        metrics['eval/' + key + '_std'] = jnp.sqrt(avg_sqr - avg ** 2)
    writer.write_scalars(step, metrics)
    return metrics['eval/hellinger_dist']


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = _CONFIG.value
  config.data_folder = os.path.join(FLAGS.data_root, config.data_folder)
  global_key = jax.random.PRNGKey(FLAGS.seed)
  global_key, model_key = jax.random.split(global_key, 2)
  writer = train_eval.setup_logging(config)
  train_ds = utils.numpy_iter(data_loader.get_dataloader(config, 'train'))
  test_tf_ds = data_loader.get_dataloader(config, 'test')
  model = PianoModel(config)
  state = model.init_state(model_key)
  fn_eval = functools.partial(
      model.eval_loop, test_tf_ds=test_tf_ds, writer=writer)
  if config.phase == 'train':
    train_eval.train_loop(
        config, writer=writer, global_key=global_key,
        train_step_fn=model.step_fn, train_ds=train_ds, state=state,
        fn_eval=fn_eval)
  else:
    assert os.path.exists(config.model_init_folder)
    train_eval.eval_latest_model(
        config.model_init_folder, writer=writer, global_key=global_key,
        state=state, fn_eval=fn_eval,
        prefix=config.get('eval_prefix', 'checkpoint_'))
  # Wait until all processes are done before exiting.
  jax.pmap(jax.random.PRNGKey)(jnp.arange(
      jax.local_device_count())).block_until_ready()


if __name__ == '__main__':
  app.run(main)
