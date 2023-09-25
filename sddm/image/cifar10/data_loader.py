"""Cifar10 dataloader."""

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from sddm.common import utils
from tensorflow_datasets.core.utils import gcs_utils




def get_dataloader(config, phase):
  """Get cifar10 data loader."""
  gcs_utils._is_gcs_disabled = True
  is_training = phase == 'train'
  dataset = tfds.load(
      'cifar10',
      split=phase,
      shuffle_files=is_training,data_dir='/media/data2/zhangpz/Code/2023/VQ-VAE-master/dataset/cifar10/cifar-10-batches-py', try_gcs=False)
  num_shards = jax.process_count()
  shard_id = jax.process_index()
  dataset = dataset.shard(num_shards=num_shards, index=shard_id)
  if is_training:
    dataset = dataset.repeat().shuffle(buffer_size=50000,
                                       seed=shard_id)
  def preprocess(x):
    """Preprocess img."""
    img = tf.cast(x['image'], tf.float32)
    aug = None
    if config.data_aug:
      if config.rand_flip:
        augment_img = tf.image.flip_left_right(img)
        aug = tf.random.uniform(shape=[]) > 0.5
        img = tf.where(aug, augment_img, img)
      if config.rot90:
        u = tf.random.uniform(shape=[])
        k = tf.cast(tf.floor(4. * u), tf.int32)
        img = tf.image.rot90(img, k=k)
        aug = aug | (k > 0)
    if aug is None:
      aug = tf.convert_to_tensor(False, dtype=tf.bool)
    out = tf.cast(img, tf.int32)
    return out

  dataset = dataset.map(preprocess,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  proc_batch_size = utils.get_per_process_batch_size(config.batch_size)
  dataset = dataset.batch(proc_batch_size // jax.local_device_count(),
                          drop_remainder=is_training)
  dataset = dataset.batch(jax.local_device_count(), drop_remainder=is_training)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset
