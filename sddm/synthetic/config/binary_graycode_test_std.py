"""Config file."""

from ml_collections import config_dict


def common_config():

  return dict(
      data_folder='synthetic/checkerboard',
      seed=1023,
      batch_size=128,
      total_train_steps=300000,
      learning_rate=1e-4,
      time_scale_factor=1000,
      time_duration=1.0,
      ema_decay=0.9999,
      lr_schedule='constant',
      diffuse_type='uniform_ot',
      optimizer='adamw',
      transformer_norm_type='prenorm',
      uniform_rate_const=1.0,
      embed_dim=512,
      num_layers=2,
      log_every_steps=50,
      plot_every_steps=50000,
      save_every_steps=10000,
      plot_samples=4096,
      eval_rounds=10,
      sampling_steps=1000,
      phase='train',
      save_root='',
      model_init_folder='',
      dtype='float32',
  )

def get_config():
  """Get config_dict."""
  cfg_dict = common_config()
  cfg_dict.update(dict(
      model_type='ebm',
      net_arch='mlp',
      embed_dim=256,
      num_layers=3,
      grad_norm=5.0,
      plot_num_batches=32,
      weight_decay=1e-6,
      sampler_type='lbjf',
      logit_type='direct',
      lambda_t='const',
      t_sample='linear',
  ))
  config = config_dict.ConfigDict(initial_dictionary=cfg_dict)
  return config
