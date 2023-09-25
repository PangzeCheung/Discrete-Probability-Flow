"""Energy based models."""

from typing import Any
from flax import linen as nn
import jax
import jax.numpy as jnp
from sddm.model import backward_model
from sddm.model import forward_model
from sddm.model import nets


class BinaryMLPScoreFunc(nn.Module):
  """Get a scalar score for an input."""
  num_layers: int
  hidden_size: int
  time_scale_factor: float = 1000.0

  @nn.compact
  def __call__(self, x, t):
    temb = nets.transformer_timestep_embedding(
        t * self.time_scale_factor, self.hidden_size)
    x = x.astype(jnp.float32)
    for _ in range(self.num_layers):
      x = nn.Dense(self.hidden_size)(x) + temb
      x = nn.elu(x)
    x = nn.Dense(1)(x)
    return x


class BinaryTransformerScoreFunc(nn.Module):
  """Get a scalar score for an input."""
  config: Any

  @nn.compact
  def __call__(self, x, t):
    config = self.config
    temb = nets.transformer_timestep_embedding(
        t * config.time_scale_factor, config.embed_dim)
    transformer = nets.MaskedTransformer(config)
    x = jnp.reshape(x, [x.shape[0], -1]).astype(jnp.int32)
    cls_token = jnp.ones((x.shape[0], 1), dtype=jnp.int32) * config.vocab_size
    x = jnp.concatenate([cls_token, x], axis=1)
    score = transformer(x, temb, 0)[..., 0]
    return score


class CatMLPScoreFunc(nn.Module):
  """Get a scalar score for an input."""
  vocab_size: int
  cat_embed_size: int
  num_layers: int
  hidden_size: int
  time_scale_factor: float = 1000.0

  @nn.compact
  def __call__(self, x, t):
    temb = nets.transformer_timestep_embedding(
        t * self.time_scale_factor, self.hidden_size)
    x = nn.Embed(self.vocab_size, self.cat_embed_size)(x)
    x = jnp.reshape(x, [x.shape[0], -1])
    for _ in range(self.num_layers):
      x = nn.Dense(self.hidden_size)(x) + temb
      x = nn.silu(x)
    x = nn.Dense(1)(x)
    return x


class BinaryScoreModel(backward_model.BackwardModel):
  """EBM for binary data."""

  def __init__(self, config):
    super(BinaryScoreModel, self).__init__(config)
    if config.net_arch == 'mlp':
      self.net = BinaryMLPScoreFunc(
          num_layers=config.num_layers, hidden_size=config.embed_dim,
          time_scale_factor=config.time_scale_factor)
    elif config.net_arch == 'transformer':
      self.net = BinaryTransformerScoreFunc(config)
    else:
      raise ValueError('Unknown net arch: %s' % config.net_arch)
    self.fwd_model = forward_model.get_fwd_model(self.config)

  def get_q(self, params, xt, t):
    """Get ll."""
    bsize = xt.shape[0]
    ddim = self.config.discrete_dim
    qxt = self.net.apply({'params': params}, x=xt, t=t)
    mask = jnp.eye(ddim).repeat(bsize, axis=0)
    xrep = jnp.tile(xt, (ddim, 1))
    xneg = (mask - xrep) * mask + (1 - mask) * xrep
    t = jnp.tile(t, (ddim,))
    qxneg = self.net.apply({'params': params}, x=xneg, t=t)
    qxt = jnp.tile(qxt, (ddim, 1))
    return qxneg, qxt

  def get_logits(self, params, xt, t):
    bsize = xt.shape[0]
    qxneg, qxt = self.get_q(params, xt, t)
    qxneg = jnp.reshape(qxneg, (-1, bsize)).T
    qxt = jnp.reshape(qxt, (-1, bsize)).T
    xt_onehot = jax.nn.one_hot(xt, 2)
    qxneg, qxt = jnp.expand_dims(qxneg, axis=-1), jnp.expand_dims(qxt, axis=-1)
    logits = xt_onehot * qxt + (1 - xt_onehot) * qxneg
    return logits

  def get_ratio(self, params, xt, t, xt_target=None):
    """Get flip ratio."""
    del xt_target
    qxneg, qxt = self.get_q(params, xt, t)
    bsize = xt.shape[0]
    ratio = jnp.exp(qxneg - qxt)
    return jnp.reshape(ratio, (-1, bsize)).T

  def get_logprob(self, params, xt, t, xt_target=None):
    """Get backwd ratio."""
    del xt_target
    logits = self.get_logits(params, xt, t)
    return backward_model.get_logprob_with_logits(self, xt, t, logits)

  def loss(self, params, rng, x0, xt, t):
    del x0, rng
    _, ll_xt = self.get_logprob(params, xt, t)
    loss = -ll_xt
    loss = jnp.sum(loss) / xt.shape[0]
    aux = {'loss': loss}
    return loss, aux


class CategoricalScoreModel(backward_model.BackwardModel):
  """EBM for categorical data."""

  def __init__(self, config):
    super(CategoricalScoreModel, self).__init__(config)
    if config.net_arch == 'mlp':
      if config.vocab_size == 2:
        self.net = BinaryMLPScoreFunc(
            num_layers=config.num_layers, hidden_size=config.embed_dim,
            time_scale_factor=config.time_scale_factor)
      else:
        self.net = CatMLPScoreFunc(
            vocab_size=config.vocab_size, cat_embed_size=config.cat_embed_size,
            num_layers=config.num_layers, hidden_size=config.embed_dim,
            time_scale_factor=config.time_scale_factor)
    else:
      raise ValueError('Unknown net arch: %s' % config.net_arch)

  def get_logits(self, params, xt, t):
    assert xt.ndim == 2
    bsize = xt.shape[0]
    ddim = self.config.discrete_dim
    vocab_size = self.config.vocab_size
    mask = jnp.eye(ddim, dtype=jnp.int32).repeat(bsize * vocab_size, axis=0)
    xrep = jnp.tile(xt, (ddim * vocab_size, 1))
    candidate = jnp.arange(vocab_size).repeat(bsize, axis=0)
    candidate = jnp.tile(jnp.expand_dims(candidate, axis=1), ((ddim, 1)))
    xall = mask * candidate + (1 - mask) * xrep
    t = jnp.tile(t, (ddim * vocab_size,))
    qall = self.net.apply({'params': params}, x=xall, t=t)
    logits = jnp.reshape(qall, (ddim, vocab_size, bsize))
    logits = jnp.transpose(logits, (2, 0, 1))
    return logits

  def get_logprob(self, params, xt, t, xt_target=None):
    bsize = xt.shape[0]
    ddim = self.config.discrete_dim
    logits = self.get_logits(params, xt, t)
    ll_all = jax.nn.log_softmax(logits, axis=-1)
    ll_xt = ll_all[jnp.arange(bsize)[:, None],
                   jnp.arange(ddim)[None, :],
                   xt]
    return ll_all, ll_xt

  def loss(self, params, rng, x0, xt, t):
    del x0, rng
    _, ll_xt = self.get_logprob(params, xt, t)
    loss = -jnp.sum(ll_xt, axis=-1)
    loss = jnp.mean(loss)
    aux = {'loss': loss}
    return loss, aux
