"""Hollow networks."""

from typing import Any
from flax import linen as nn
import jax
import jax.numpy as jnp
from sddm.model import backward_model
from sddm.model import nets


def bidir_transformer(config, x, temb, readout_dim=None):
  """Bidirectional Transformer procedure."""

  if readout_dim is None:
    readout_dim = config.vocab_size
  input_shape = list(x.shape)[:-1]
  x = jnp.reshape(x, [x.shape[0], -1, x.shape[-1]])
  if config.net_arch == 'bidir_transformer':
    module = nets.UniDirectionalTransformer
  elif config.net_arch == 'bidir_combiner_transformer':
    module = nets.CombinerAxial
  else:
    raise ValueError('Unknown net_arch: %s' % config.net_arch)
  l2r_embed = module(config, 'l2r')(x, temb)
  r2l_embed = module(config, 'r2l')(x, temb)
  if config.bidir_readout == 'concat':
    readout_module = nets.ConcatReadout
  elif config.bidir_readout == 'res_concat':
    readout_module = nets.ConcatResidualReadout
  elif config.bidir_readout == 'attention':
    readout_module = nets.AttentionReadout
  else:
    raise ValueError('Unknown bidir_readout: %s' % config.bidir_readout)
  logits = readout_module(config, readout_dim=readout_dim)(
      l2r_embed, r2l_embed, temb)
  logits = jnp.reshape(logits, input_shape + [readout_dim])
  return logits


class BidirectionalTransformer(nn.Module):
  """Transformer in two directions."""

  config: Any

  @nn.compact
  def __call__(self, x, t):
    config = self.config
    x = nn.Embed(config.vocab_size, config.embed_dim)(x)
    temb = nets.transformer_timestep_embedding(
        t * config.time_scale_factor, config.embed_dim)
    return bidir_transformer(config, x, temb)


class EnumerativeTransformer(nn.Module):
  """Enumerative transformer."""

  config: Any

  @nn.compact
  def __call__(self, x, t):
    config = self.config
    temb = nets.transformer_timestep_embedding(
        t * config.time_scale_factor, config.embed_dim)
    transformer = nets.MaskedTransformer(config)
    x_shape = x.shape
    x = jnp.reshape(x, [x.shape[0], -1])

    def masked_logits(pos):
      x_masked = x.at[:, pos].set(config.vocab_size)
      logits = transformer(x_masked, temb, pos)
      logits = jnp.squeeze(logits, axis=1)
      return logits

    prefix_cond = config.get('conditional_dim', 0)
    logits = jax.vmap(masked_logits, out_axes=1)(
        jnp.arange(prefix_cond, x.shape[1]))
    if prefix_cond:
      dummy_logits = jnp.zeros(
          [x.shape[0], prefix_cond] + list(logits.shape[2:]), dtype=jnp.float32)
      logits = jnp.concatenate([dummy_logits, logits], axis=1)
    logits = jnp.reshape(logits, list(x_shape) + [config.vocab_size])
    return logits


def prefix_conditional_forward(x, t, config, net_fn):
  """Logits prediction with prefix conditioning."""
  x = nn.Embed(config.vocab_size, config.embed_dim)(x)
  temb = nets.transformer_timestep_embedding(
      t * config.time_scale_factor, config.embed_dim)
  conditioner, x = jnp.split(x, [config.conditional_dim], axis=1)
  logits = net_fn(x, temb, conditioner)
  dummy_logits = jnp.zeros(
      [x.shape[0], config.conditional_dim] + list(logits.shape[2:]),
      dtype=jnp.float32)
  logits = jnp.concatenate([dummy_logits, logits], axis=1)
  assert logits.shape[1] == config.conditional_dim + x.shape[1]
  return logits


class PrefixConditionalBidirTransformer(nn.Module):
  """Transformer in two directions with prefix conditioning."""

  config: Any

  @nn.compact
  def __call__(self, x, t):
    config = self.config

    def logits_fn(x, temb, conditioner):
      if config.net_arch == 'bidir_transformer':
        module = nets.UniDirectionalTransformer
      elif config.net_arch == 'bidir_combiner_transformer':
        module = nets.CombinerAxial
      else:
        raise ValueError('Unknown net_arch: %s' % config.net_arch)
      l2r_embed = module(config, 'l2r')(x, temb, conditioner)[:, -x.shape[1]:]
      r2l_embed = module(config, 'r2l')(x, temb, conditioner)[:, :x.shape[1]]
      if config.bidir_readout == 'concat':
        readout_module = nets.ConcatReadout
      elif config.bidir_readout == 'res_concat':
        readout_module = nets.ConcatResidualReadout
      elif config.bidir_readout == 'attn':
        readout_module = nets.AttentionReadout
      else:
        raise ValueError('Unknown bidir_readout: %s' % config.bidir_readout)
      logits = readout_module(config)(l2r_embed, r2l_embed, temb)
      return logits

    return prefix_conditional_forward(x, t, config, logits_fn)


class HollowModel(backward_model.CondFactorizedBackwardModel):
  """Hollow model for discrete data."""

  def __init__(self, config):
    super(HollowModel, self).__init__(config)
    if 'bidir' in config.net_arch and 'transformer' in config.net_arch:
      self.net = BidirectionalTransformer(config)
    elif config.net_arch == 'enum_transformer':
      self.net = EnumerativeTransformer(config)
    else:
      raise ValueError('Unknown net arch: %s' % config.net_arch)


class PrefixCondHollowModel(HollowModel):
  """Hollow model for discrete data with prefix conditioning."""

  def __init__(self, config):
    super(PrefixCondHollowModel, self).__init__(config)
    if 'bidir' in config.net_arch and 'transformer' in config.net_arch:
      self.net = PrefixConditionalBidirTransformer(config)
    elif config.net_arch == 'enum_transformer':
      self.net = EnumerativeTransformer(config)
    else:
      raise ValueError('Unknown net arch: %s' % config.net_arch)

  def loss(self, params, rng, x0, xt, t):
    del x0, rng
    ll_all, log_xt = self.get_logprob(params, xt, t)
    ll_all = ll_all[:, self.config.conditional_dim:]
    log_xt = log_xt[:, self.config.conditional_dim:]
    loss = self.calc_loss(xt, t, ll_all, log_xt)
    loss = jnp.sum(loss) / xt.shape[0]
    aux = {'loss': loss}
    return loss, aux
