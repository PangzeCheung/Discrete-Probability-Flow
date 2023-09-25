"""Common train/eval."""

import os
from absl import logging
from clu import metric_writers
import flax
from flax import jax_utils
from flax.training import checkpoints
import jax
import jax.numpy as jnp
from sddm.common import utils


def setup_logging(config):
  """Setup logging and writer."""
  if jax.process_index() == 0:
    logging.info(config)
    logging.info('process count: %d', jax.process_count())
    logging.info('device count: %d', jax.device_count())
    logging.info('device/host: %d', jax.local_device_count())

  writer = metric_writers.create_default_writer(
      config.save_root, just_logging=jax.process_index() > 0)
  if jax.process_index() == 0:
    fig_folder = os.path.join(config.save_root, 'figures')
    if not os.path.exists(fig_folder):
      os.makedirs(fig_folder)
    config.fig_folder = fig_folder
  return writer


def eval_latest_model(folder, writer, global_key, state, fn_eval,
                      prefix='checkpoint_'):
  state = checkpoints.restore_checkpoint(os.path.join(folder,'ckpts'), state, prefix=prefix)
  loaded_step = state.step
  logging.info('Restored from %s at step %d', folder, loaded_step)
  state = flax.jax_utils.replicate(state)
  process_rng_key = jax.random.fold_in(global_key, jax.process_index())

  with metric_writers.ensure_flushes(writer):
    metric = fn_eval(loaded_step, state, process_rng_key)

  print(metric)

def eval_variance_latest_model(folder, writer, global_key, state, fn_eval,
                      prefix='checkpoint_'):
  state = checkpoints.restore_checkpoint(os.path.join(folder,'ckpts'), state, prefix=prefix)
  loaded_step = state.step
  logging.info('Restored from %s at step %d', folder, loaded_step)
  state = flax.jax_utils.replicate(state)
  process_rng_key = jax.random.fold_in(global_key, jax.process_index())


  fn_eval(loaded_step, state, process_rng_key)



def train_loop(config, writer, global_key, state, train_ds,
               train_step_fn, fn_plot_data=None, fn_eval=None,
               fn_data_preprocess=None):
  """Train loop."""
  if os.path.exists(config.get('model_init_folder', '')):
    state = checkpoints.restore_checkpoint(
        config.model_init_folder, state)
    logging.info('Restored from %s at step %d',
                 config.model_init_folder, state.step)
  ckpt_folder = os.path.join(config.save_root, 'ckpts')
  if os.path.exists(ckpt_folder):
    state = checkpoints.restore_checkpoint(ckpt_folder, state)
    logging.info('Restored from %s at step %d', ckpt_folder, state.step)
  init_step = state.step
  state = flax.jax_utils.replicate(state)
  process_rng_key = jax.random.fold_in(global_key, jax.process_index())
  train_step_fn = jax.pmap(train_step_fn, axis_name='shard')
  lr_schedule = utils.build_lr_schedule(config)
  best_metric = None
  if fn_data_preprocess is None:
    fn_data_preprocess = lambda x: x
  def save_model(state, step, prefix='checkpoint_', overwrite=False):
    if jax.process_index() == 0:
      host_state = jax.device_get(jax_utils.unreplicate(state))
      checkpoints.save_checkpoint(ckpt_folder, host_state, step,
                                  prefix=prefix, overwrite=overwrite, keep_every_n_steps = 20000) #, keep_every_n_steps=10000

  with metric_writers.ensure_flushes(writer):
    num_params = sum(x.size for x in jax.tree_leaves(state.params))
    writer.write_scalars(0, {'num_params': num_params})
    for step in range(init_step + 1, config.total_train_steps + 1):
      batch = fn_data_preprocess(next(train_ds))
      process_rng_key = jax.random.fold_in(process_rng_key, step)
      step_rng_keys = utils.shard_prng_key(process_rng_key)
      state, aux = train_step_fn(state, step_rng_keys, batch)
      if step % config.log_every_steps == 0:
        aux = jax.device_get(flax.jax_utils.unreplicate(aux))
        aux['train/lr'] = lr_schedule(step)
        writer.write_scalars(step, aux)
      if step % config.plot_every_steps == 0 and fn_eval is not None:
        metric = fn_eval(step, state, process_rng_key)
        if metric is not None:
          if best_metric is None or metric < best_metric:
            best_metric = metric
            save_model(state, step, prefix='bestckpt_', overwrite=True)

      if step % config.save_every_steps == 0:
        save_model(state, step)

