"""Dump synthetic data into numpy array."""

from collections.abc import Sequence
import os
from absl import app
from absl import flags
from ml_collections import config_flags
import numpy as np
import tqdm
from sddm.synthetic.data import utils

_CONFIG = config_flags.DEFINE_config_file('data_config', lock_config=False)
flags.DEFINE_integer('num_samples', 10000000, 'num samples to be generated')
flags.DEFINE_integer('batch_size', 200, 'batch size for datagen')
flags.DEFINE_string('data_root', None, 'root folder of data')

FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if not os.path.exists(FLAGS.data_root):
    os.makedirs(FLAGS.data_root)
  data_config = _CONFIG.value
  if data_config.vocab_size == 2:
    db, bm, inv_bm = utils.setup_data(data_config)

    with open(os.path.join(FLAGS.data_root, 'config.yaml'), 'w') as f:
      f.write(data_config.to_yaml())
    data_list = []
    for _ in tqdm.tqdm(range(FLAGS.num_samples // FLAGS.batch_size)):
      data = utils.float2bin(db.gen_batch(FLAGS.batch_size), bm,
                             data_config.discrete_dim, data_config.int_scale)
      data_list.append(data.astype(bool))
    data = np.concatenate(data_list, axis=0)
    print(data.shape[0], 'samples generated')
    save_path = os.path.join(FLAGS.data_root, 'data.npy')
    with open(save_path, 'wb') as f:
      np.save(f, data)

    with open(os.path.join(FLAGS.data_root, 'samples.pdf'), 'wb') as f:
      float_data = utils.bin2float(data[:1000].astype(np.int32), inv_bm,
                                   data_config.discrete_dim,
                                   data_config.int_scale)
      utils.plot_samples(float_data, f, im_size=4.1, im_fmt='pdf')
  elif data_config.vocab_size <= 10:
    db = utils.our_setup_data(data_config)
    with open(os.path.join(FLAGS.data_root, 'config.yaml'), 'w') as f:
      f.write(data_config.to_yaml())
    data_list = []
    for _ in tqdm.tqdm(range(FLAGS.num_samples // FLAGS.batch_size)):
      data = utils.ourfloat2base(db.gen_batch(FLAGS.batch_size),
                             data_config.discrete_dim, data_config.f_scale, data_config.int_scale, data_config.vocab_size)
      data_list.append(data)
    data = np.concatenate(data_list, axis=0)
    print(data.shape[0], 'samples generated')
    save_path = os.path.join(FLAGS.data_root, 'data.npy')

    with open(save_path, 'wb') as f:
      np.save(f, data)

    with open(os.path.join(FLAGS.data_root, 'samples.pdf'), 'wb') as f:
      float_data = utils.ourbase2float(data[:1000].astype(np.int32),
                                   data_config.discrete_dim,
                                   data_config.f_scale,
                                   data_config.int_scale,
                                   data_config.vocab_size)
      utils.plot_samples(float_data, f, im_size=4.1, im_fmt='pdf')

if __name__ == '__main__':
  app.run(main)
