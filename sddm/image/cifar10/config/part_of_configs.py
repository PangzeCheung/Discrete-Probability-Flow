from ml_collections import config_dict
def common_config():
  sweep = [{'config.diffuse_type': 'uniform',
            'config.sampler_type': 'lbjf',
            'config.sampling_steps': 1000}]

  return dict(
      launch=dict(
          name='cifar10_hollow_vq_train',
          priority=200,
          resource={
              'pf': '4x4x4',
          },
          build_target_path='//experimental/brain/locally_balanced/diffusion/image/cifar10:main_cifar10',
          sweep=sweep,
      ),
      vq_model_info={
      },
      vq_model='Cifar10/FID514',
      data_folder='',
      seed=1023,
      data_aug=True,
      rand_flip=True,
      rot90=False,
      batch_size=32,
      learning_rate=1e-3,
      time_scale_factor=1000,
      corrector_frac=0.0,
      corrector_steps=0,
      corrector_scale=1.0,
      time_duration=1.0,
      ema_decay=0.9999,
      lr_schedule='constant',
      transformer_norm_type='prenorm',
      log_every_steps=50,
      plot_every_steps=5000,
      save_every_steps=5000,
      plot_samples=256,
      plot_num_batches=1,
      sampling_steps=1000,
      total_train_steps=1500000,
      phase='train',
      model_init_folder='',
      save_root='',
      dtype='float32',
  )


def get_config():
  """Get config_dict."""
  cfg_dict = common_config()
  cfg_dict.update(dict(
      model_type='vq_hollow',
      net_arch='bidir_transformer',
      lr_schedule='updown',
      warmup_frac=0.03,
      bidir_readout='res_concat',
      num_output_ffresiduals=2,
      optimizer='adamw',
      loss_type='rm',
      grad_norm=5.0,
      weight_decay=1e-6,
      logit_type='reverse_logscale',
      sampler_type='exact',
      diffuse_type='uniform',
      t_func='log_sqr',
      readout='resnet',
      is_ordinal=False,
      top_vocab=0,
      clean_vocab=False,
      uniform_rate_const=0.035,
      num_heads=12,
      num_layers=12,
      embed_dim=768,
      qkv_dim=768,
      mlp_dim=3072,
      dropout_rate=0.0,
      learning_rate=1e-4,
      attention_dropout_rate=0.0,
      dropout_deterministic=False,
  ))
  config = config_dict.ConfigDict(initial_dictionary=cfg_dict)
  return config