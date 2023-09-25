"""Neural networks."""

import functools
import math
from typing import Sequence, Any, Callable, Optional
from flax import linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
  features: Sequence[int]
  activation: Any = nn.relu

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = self.activation(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x


def apply_film(film_params, x):
  film_params = jnp.expand_dims(film_params, axis=1)
  assert film_params.ndim == 3 and x.ndim == 3
  a, b = jnp.split(film_params, 2, axis=-1)
  x = a * x + b
  return x


class ConcatReadout(nn.Module):
  """Concat two directions."""

  config: Any
  readout_dim: int = 0

  @nn.compact
  def __call__(self, l2r_embed, r2l_embed, _):
    config = self.config
    state = jnp.concatenate([l2r_embed, r2l_embed], axis=-1)
    out_dim = self.readout_dim
    if out_dim == 0:
      out_dim = config.vocab_size
    predictor = MLP([2 * config.embed_dim, out_dim],
                    activation=nn.gelu)
    logits = predictor(state)
    return logits


class ResidualReadout(nn.Module):
  """Use residual net to readout logits."""

  config: Any
  readout_dim: int = 0

  @nn.compact
  def __call__(self, x, temb):
    config = self.config
    embed_dim = x.shape[-1]
    temb = MLP([config.mlp_dim, 4 * temb.shape[1]], activation=nn.gelu)(temb)
    for _ in range(config.num_output_ffresiduals):
      film_params = nn.Dense(2 * embed_dim)(temb)
      z = MLP([config.mlp_dim, embed_dim], activation=nn.gelu)(x)
      x = nn.LayerNorm(dtype=config.dtype)(x + z)
      x = apply_film(film_params, x)
    out_dim = self.readout_dim
    if out_dim == 0:
      out_dim = config.vocab_size
    logits = nn.Dense(out_dim)(x)
    return logits


class ConcatResidualReadout(nn.Module):
  """Concat two directions and use residual net."""

  config: Any
  readout_dim: int = 0

  @nn.compact
  def __call__(self, l2r_embed, r2l_embed, temb):
    config = self.config
    state = jnp.concatenate([l2r_embed, r2l_embed], axis=-1)
    return ResidualReadout(config, readout_dim=self.readout_dim)(state, temb)


# From https://github.com/yang-song/score_sde_pytorch/ which is from
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
def transformer_timestep_embedding(timesteps, embedding_dim,
                                   max_positions=10000):
  """Get time embedding for timesteps."""
  assert embedding_dim % 2 == 0
  assert len(timesteps.shape) == 1
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
  emb = timesteps[:, None] * emb[None, :]
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


class TransformerMlpBlock(nn.Module):
  """Transformer MLP block."""

  mlp_dim: int
  out_dim: Optional[int] = None
  dtype: Any = jnp.float32
  dropout_rate: float = 0.0
  dropout_deterministic: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = (inputs.shape[-1] if self.out_dim is None
                      else self.out_dim)
    x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(
            inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=self.dropout_rate)(
        x, deterministic=self.dropout_deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(
            x)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=self.dropout_deterministic)
    return output


def sa_block(config, inputs, masks):
  """Self-attention block."""
  if config.transformer_norm_type == 'prenorm':
    x = nn.LayerNorm(dtype=config.dtype)(inputs)
    x = nn.SelfAttention(
        num_heads=config.num_heads,
        dtype=config.dtype,
        qkv_features=config.qkv_dim,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
        deterministic=config.dropout_deterministic,
        decode=False)(x, masks)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=config.dropout_deterministic)
    x = x + inputs
  elif config.transformer_norm_type == 'postnorm':
    x = nn.SelfAttention(
        num_heads=config.num_heads,
        dtype=config.dtype,
        qkv_features=config.qkv_dim,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
        deterministic=config.dropout_deterministic,
        decode=False)(inputs, masks)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=config.dropout_deterministic)
    x = x + inputs
    x = nn.LayerNorm(dtype=config.dtype)(x)
  else:
    raise ValueError('unknown norm type %s' % config.transformer_norm_type)
  return x


def cross_attention(config, l2r_embed, r2l_embed, temb):
  """Cross attention to both directions."""
  seq_len = l2r_embed.shape[1]
  temb = jnp.expand_dims(temb, axis=1)
  head_dim = config.qkv_dim // config.num_heads
  dense = functools.partial(nn.linear.DenseGeneral,
                            axis=-1,
                            features=(config.num_heads, head_dim))
  query = dense(name='query')(l2r_embed + r2l_embed)
  all_embed = jnp.concatenate([temb, l2r_embed, r2l_embed], axis=1)
  key = dense(name='key')(all_embed)
  val = dense(name='val')(all_embed)
  query = query / jnp.sqrt(query.shape[-1]).astype(config.dtype)
  logits = jnp.einsum('bqhd,bkhd->bhqk', query, key)

  idx = jnp.arange(seq_len, dtype=jnp.int32)
  att_l2r_mask = nn.attention.make_attention_mask(idx, idx, jnp.greater_equal)
  att_r2l_mask = nn.attention.make_attention_mask(idx, idx, jnp.less_equal)
  att_t = jnp.ones((1, seq_len, 1))
  joint_mask = jnp.concatenate([att_t, att_l2r_mask, att_r2l_mask], axis=-1)
  joint_mask = jnp.expand_dims(joint_mask, axis=0)
  attn_weights = jnp.where(joint_mask, logits, jnp.finfo(config.dtype).min)
  attn_weights = jax.nn.softmax(attn_weights, axis=-1)
  x = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, val)
  x = nn.linear.DenseGeneral(
      features=(config.embed_dim,),
      axis=(-2, -1),
      name='out'
  )(x)
  return x


class AttentionReadout(nn.Module):
  """Attention readout of bidir embed."""

  config: Any
  readout_dim: int = 0

  @nn.compact
  def __call__(self, l2r_embed, r2l_embed, temb):
    config = self.config
    inputs = l2r_embed + r2l_embed
    if config.transformer_norm_type == 'prenorm':
      l2r_embed = nn.LayerNorm(dtype=config.dtype)(l2r_embed)
      r2l_embed = nn.LayerNorm(dtype=config.dtype)(r2l_embed)
      x = cross_attention(config, l2r_embed, r2l_embed, temb)
      x = x + inputs
    elif config.transformer_norm_type == 'postnorm':
      x = cross_attention(config, l2r_embed, r2l_embed, temb)
      x = x + inputs
      x = nn.LayerNorm(dtype=config.dtype)(x)
    else:
      raise ValueError('unknown norm type %s' % config.transformer_norm_type)
    x = ff_block(config, x)
    return ResidualReadout(config, self.readout_dim)(x, temb)


def ff_block(config, x):
  """Feed-forward block."""
  if config.transformer_norm_type == 'prenorm':
    z = nn.LayerNorm(dtype=config.dtype)(x)
    z = TransformerMlpBlock(
        mlp_dim=config.mlp_dim,
        dtype=config.dtype,
        dropout_rate=config.dropout_rate,
        dropout_deterministic=config.dropout_deterministic)(z)
    z = x + z
  elif config.transformer_norm_type == 'postnorm':
    z = TransformerMlpBlock(
        mlp_dim=config.mlp_dim,
        dtype=config.dtype,
        dropout_rate=config.dropout_rate,
        dropout_deterministic=config.dropout_deterministic)(x)
    z = x + z
    z = nn.LayerNorm(dtype=config.dtype)(z)
  else:
    raise ValueError('unknown norm type %s' % config.transformer_norm_type)
  return z


class TransformerBlock(nn.Module):
  """Transformer block."""

  config: Any

  @nn.compact
  def __call__(self, inputs, masks):
    assert inputs.ndim == 3
    config = self.config
    x = sa_block(config, inputs, masks)
    # MLP block.
    z = ff_block(config, x)
    return z


class TransformerEncoder(nn.Module):
  """Transformer encoder."""

  config: Any

  @nn.compact
  def __call__(self, x, temb, conditioner=None):
    assert x.ndim == 3 and temb.ndim == 2
    config = self.config
    temb = jnp.expand_dims(temb, axis=1)
    if conditioner is None:
      conditioner = temb
    else:
      conditioner = jnp.concatenate([conditioner, temb], axis=1)
    x = jnp.concatenate([conditioner, x], axis=1)
    pos_embed = self.param(
        'pos_embed', nn.initializers.xavier_uniform(),
        (1, x.shape[1], x.shape[2]), config.dtype,
    )
    x = x + pos_embed
    x = nn.Dropout(config.dropout_rate)(
        x, deterministic=config.dropout_deterministic)
    for layer_idx in range(config.num_layers):
      x = TransformerBlock(
          name='block_{}'.format(layer_idx),
          config=config)(x, masks=None)
    x = x[:, 1:]
    return x


class MaskedTransformer(nn.Module):
  """Masked transformer."""

  config: Any

  @nn.compact
  def __call__(self, x, temb, pos):
    config = self.config
    x = nn.Embed(config.vocab_size + 1, config.embed_dim)(x)
    embed = TransformerEncoder(config)(x, temb)
    embed = jnp.expand_dims(embed[:, pos], axis=1)
    if config.readout == 'mlp':
      logits = MLP([2 * config.embed_dim, config.vocab_size],
                   activation=nn.gelu)(embed)
    elif config.readout == 'resnet':
      logits = ResidualReadout(config)(embed, temb)
    else:
      raise ValueError('Unknown readout type %s' % config.readout)
    return logits


class UniDirectionalTransformer(nn.Module):
  """Transformer in one direction."""

  config: Any
  direction: str

  @nn.compact
  def __call__(self, x, temb, conditioner=None):
    assert x.ndim == 3 and temb.ndim == 2
    temb = jnp.expand_dims(temb, axis=1)
    if conditioner is None:
      conditioner = temb
    else:
      conditioner = jnp.concatenate([conditioner, temb], axis=1)
    config = self.config
    cond_dim = conditioner.shape[1]
    concat_dim = x.shape[1] + cond_dim - 1
    pos_idx = jnp.expand_dims(jnp.arange(concat_dim, dtype=jnp.int32), 0)
    if self.direction == 'l2r':
      x = jnp.concatenate([conditioner, x[:, :-1]], axis=1)
      mask = nn.attention.make_attention_mask(pos_idx, pos_idx,
                                              jnp.greater_equal)
      mask = mask.at[:, :, :cond_dim, :cond_dim].set(1.0)
    else:
      x = jnp.concatenate([x[:, 1:], conditioner], axis=1)
      mask = nn.attention.make_attention_mask(pos_idx, pos_idx,
                                              jnp.less_equal)
      mask = mask.at[:, :, -cond_dim:, -cond_dim:].set(1.0)
    pos_embed = self.param(
        'pos_embed', nn.initializers.xavier_uniform(),
        (1, concat_dim, x.shape[2]), config.dtype,
    )
    x = x + pos_embed
    x = nn.Dropout(config.dropout_rate)(
        x, deterministic=config.dropout_deterministic)
    for layer_idx in range(config.num_layers):
      x = TransformerBlock(
          name='block_{}'.format(layer_idx),
          config=config)(x, masks=mask)
    return x
