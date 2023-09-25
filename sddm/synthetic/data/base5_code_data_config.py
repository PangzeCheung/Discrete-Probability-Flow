"""Data config file."""

from ml_collections import config_dict


def get_config():
  config = config_dict.ConfigDict()
  config.discrete_dim = 16
  config.vocab_size = 5
  config.binmode = ''
  config.data_name = ''
  config.int_scale = -1.0
  config.plot_size = -1.0
  config.f_scale = -1.0
  return config
