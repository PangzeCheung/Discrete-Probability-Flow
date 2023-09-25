"""TF data loder for piano data."""

import os
from absl import logging
import jax
import numpy as np
import tensorflow as tf
from sddm.common import utils


def get_dataloader(config, phase):
  """Get piano data loader."""

  data_file = os.path.join(config.data_folder, '%s.npy' % phase)
  with open(data_file, 'rb') as f:
    data = np.load(f)
  is_training = phase == 'train'
  if not is_training:
    if data.shape[0] % config.batch_size != 0:
      pad_size = config.batch_size - data.shape[0] % config.batch_size
    else:
      pad_size = 0
    mask = np.array([1] * data.shape[0] + [0] * pad_size, dtype=data.dtype)
    data = np.concatenate(
        [data, np.zeros((pad_size, data.shape[1]), dtype=data.dtype)], axis=0)
    data = np.concatenate([mask[:, None], data], axis=1)
  logging.info('data shape: %s', str(data.shape))
  num_shards = jax.process_count()
  shard_id = jax.process_index()
  dataset = tf.data.Dataset.from_tensor_slices(data)
  dataset = dataset.shard(num_shards=num_shards, index=shard_id)
  if is_training:
    dataset = dataset.repeat().shuffle(buffer_size=data.shape[0],
                                       seed=shard_id)
  dataset = dataset.map(lambda x: tf.cast(x, tf.int32),
                        num_parallel_calls=tf.data.AUTOTUNE)
  proc_batch_size = utils.get_per_process_batch_size(config.batch_size)
  dataset = dataset.batch(proc_batch_size // jax.local_device_count(),
                          drop_remainder=is_training)
  dataset = dataset.batch(jax.local_device_count(), drop_remainder=is_training)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset
