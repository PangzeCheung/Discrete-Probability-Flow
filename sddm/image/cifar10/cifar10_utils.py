"""Cifar10 model and utils."""

import copy
import functools
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from sddm.common import utils
from sddm.image import image_diffusion
from sddm.image.cifar10 import data_loader
from sddm.model import continuous_time_diffusion as ct_diff
from sddm.model import discrete_time_diffusion as dt_diff


class Cifar10(image_diffusion.ImageDiffusion):
  """Cifar10 model."""

  def __init__(self, config):
    super(Cifar10, self).__init__(config, image_size=32)


class CtCifar10(ct_diff.CategoricalDiffusionModel, Cifar10):
  """Continuous time cifar10."""

  def __init__(self, config):
    ct_diff.CategoricalDiffusionModel.__init__(self, config)
    Cifar10.__init__(self, config)


class VqCifar10(image_diffusion.VqImageDiffusion):
  """VQ Cifar10 model."""

  def __init__(self, config):
    super(VqCifar10, self).__init__(config, image_size=32)

  #@functools.partial(jax.pmap, static_broadcasted_argnums=0)
  def encode_batch(self, batch):
    tokens = super(VqCifar10, self).encode_single_batch(batch)
    if self.config.clean_vocab:
      tokens = self.vocab_map[tokens]
    return tokens

  def decode_tokens(self, tokens):
    if self.config.clean_vocab:
      tokens = self.reverse_map[tokens]
    return super(VqCifar10, self).decode_tokens(tokens)


class CtVqCifar10(ct_diff.CategoricalDiffusionModel, VqCifar10):
  """Continuous time vq cifar10."""

  def __init__(self, config):
    ct_diff.CategoricalDiffusionModel.__init__(self, config)
    VqCifar10.__init__(self, config)

  def sample_loop(self, state, rng, num_samples=None, conditioner=None):
    tokens = ct_diff.CategoricalDiffusionModel.sample_loop(
        self, state, rng, num_samples, conditioner)
    return tokens


class D3pmVqCifar10(dt_diff.D3PM, VqCifar10):
  """D3PM vq cifar10."""

  def __init__(self, config):
    dt_diff.D3PM.__init__(self, config)
    VqCifar10.__init__(self, config)

  def sample_loop(self, state, rng, num_samples=None, conditioner=None):
    tokens = dt_diff.D3PM.sample_loop(
        self, state, rng, num_samples, conditioner)
    x0 = self.decode_tokens(tokens)
    return x0


def postprocess_config(config):
  """Process discrete dim and vocab size."""
  config.discrete_dim = (8, 8)
  config.vocab_size = 512
