"""Generic model/trainer for discrete time models, largely based on D3PM code."""

import functools
from typing import Sequence
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as onp
from sddm.common import utils
from sddm.model import backward_model
from sddm.model import continuous_time_diffusion
from sddm.model import dt_forward_model


def get_timestep_embedding(timesteps, embedding_dim: int,
                           max_time=1000., dtype=jnp.float32):
  """Build sinusoidal embeddings (from Fairseq).

  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".

  Args:
    timesteps: jnp.ndarray: generate embedding vectors at these timesteps
    embedding_dim: int: dimension of the embeddings to generate
    max_time: float: largest time input
    dtype: data type of the generated embeddings

  Returns:
    embedding vectors with shape `(len(timesteps), embedding_dim)`
  """
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  timesteps *= (1000. / max_time)

  half_dim = embedding_dim // 2
  emb = onp.log(10000) / (half_dim - 1)
  emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
  emb = timesteps.astype(dtype)[:, None] * emb[None, :]
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


class MLP(nn.Module):
  """Mlp."""

  features: Sequence[int]
  num_pixel_vals: int = 2
  max_time: float = 1000.

  @nn.compact
  def __call__(self, x, t):
    temb = get_timestep_embedding(t, self.features[0], max_time=self.max_time)
    x = x.astype(jnp.float32)
    feat_dim = 1
    for i in x.shape[1:]:
      feat_dim *= i
    input_shape = x.shape
    for feat in self.features:
      x = nn.swish(nn.Dense(feat)(x) + temb)
    out = nn.Dense(feat_dim * self.num_pixel_vals)(x)
    out_shape = input_shape + (self.num_pixel_vals,)
    out = jnp.reshape(out, out_shape)
    return out


class D3PMBackward(backward_model.BackwardModel):
  """D3PM backward model."""

  def __init__(self, config, fwd_model):
    self.config = config
    self.hybrid_coeff = config.hps.hybrid_coeff
    self.fwd_model = fwd_model
    self.eps = self.fwd_model.eps
    self.num_states = config.vocab_size
    self.backwd_gap = config.get('sampling_gap', 1)
    if config.net_arch == 'transformer':
      self.net = backward_model.FreeformTransformer(config)
    elif config.net_arch == 'mlp':
      self.net = MLP([256, 256], num_pixel_vals=config.vocab_size)
    else:
      raise ValueError('Unknown net arch: %s' % config.net_arch)

  def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
    """Compute logits of q(x_{t-gap} | x_t, x_start)."""

    if x_start_logits:
      assert x_start.shape == x_t.shape + (self.num_states,), (
          x_start.shape, x_t.shape)
    else:
      assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)

    fact1 = self.fwd_model.at_transpose_onestep(t, x_t)
    new_t = jnp.clip(t - self.backwd_gap, a_min=0)
    if x_start_logits:
      fact2 = self.fwd_model.at_onehot(new_t, jax.nn.softmax(x_start, axis=-1))
      tzero_logits = x_start
    else:
      fact2 = self.fwd_model.at(new_t, x_start)
      tzero_logits = jnp.log(
          jax.nn.one_hot(x_start, num_classes=self.num_states) + self.eps)

    # At t=0 we need the logits of q(x_{-1}|x_0, x_start)
    # where x_{-1} == x_start. This should be equal the log of x_0.
    out = jnp.log(fact1 + self.eps) + jnp.log(fact2 + self.eps)
    t_broadcast = jnp.expand_dims(t, tuple(range(1, out.ndim)))
    return jnp.where(t_broadcast < self.backwd_gap, tzero_logits, out)

  def get_logits(self, params, xt, t):
    """Compute logits of p(x_{t-1} | x_t)."""
    assert t.shape == (xt.shape[0],)
    model_output = self.net.apply({'params': params}, x=xt, t=t)
    model_logits = model_output
    # Predict the logits of p(x_{t-1}|x_t) by parameterizing this distribution
    # as ~ sum_{pred_x_start} q(x_{t-1}, x_t |pred_x_start)p(pred_x_start|x_t)
    pred_x_start_logits = model_logits
    t_broadcast = jnp.expand_dims(t, tuple(range(1, model_logits.ndim)))
    q_pos = self.q_posterior_logits(pred_x_start_logits, xt, t,
                                    x_start_logits=True)
    model_logits = jnp.where(t_broadcast == 0, pred_x_start_logits, q_pos)
    assert (model_logits.shape ==
            pred_x_start_logits.shape == xt.shape + (self.num_states,))
    return model_logits, pred_x_start_logits

  def vb_terms_bpd(self, params, *, x_start, xt, t):
    """Calculate specified terms of the variational bound.

    Args:
      params: parameters
      x_start: original clean data
      xt: noisy data
      t: timestep of the noisy data (and the corresponding term of the bound
        to return)

    Returns:
      a pair `(kl, pred_start_logits)`, where `kl` are the requested bound terms
      (specified by `t`), and `pred_x_start_logits` is logits of
      the denoised image.
    """
    true_logits = self.q_posterior_logits(x_start, xt, t, x_start_logits=False)
    model_logits, pred_x_start_logits = self.get_logits(params, xt=xt, t=t)

    kl = utils.categorical_kl_logits(logits1=true_logits, logits2=model_logits)
    assert kl.shape == x_start.shape
    kl = utils.meanflat(kl) / onp.log(2.)

    decoder_nll = -utils.categorical_log_likelihood(x_start, model_logits)
    assert decoder_nll.shape == x_start.shape
    decoder_nll = utils.meanflat(decoder_nll) / onp.log(2.)

    # At the first timestep return the decoder NLL,
    # otherwise return KL(q(x_{t-1}|x_t,x_start) || p(x_{t-1}|x_t))
    assert kl.shape == decoder_nll.shape == t.shape == (x_start.shape[0],)
    return jnp.where(t == 0, decoder_nll, kl), pred_x_start_logits

  def cross_entropy_x_start(self, x_start, pred_x_start_logits):
    """Calculate crossentropy between x_start and predicted x_start.

    Args:
      x_start: original clean data
      pred_x_start_logits: predicted_logits

    Returns:
      ce: cross entropy.
    """

    ce = -utils.categorical_log_likelihood(x_start, pred_x_start_logits)
    assert ce.shape == x_start.shape
    ce = utils.meanflat(ce) / onp.log(2.)

    assert ce.shape == (x_start.shape[0],)

    return ce

  def loss(self, params, rng, x0, xt, t):
    del rng
    vb_losses, pred_x_start_logits = self.vb_terms_bpd(
        params=params, x_start=x0, xt=xt, t=t)
    ce_losses = self.cross_entropy_x_start(
        x_start=x0, pred_x_start_logits=pred_x_start_logits)
    losses = vb_losses + self.hybrid_coeff * ce_losses
    loss = jnp.mean(losses)
    aux = {'loss': loss}
    return loss, aux


class D3PM(continuous_time_diffusion.DiffusionModel):
  """Discrete time Model interface."""

  def __init__(self, config):
    self.config = config
    self.optimizer = utils.build_optimizer(config)
    self.fwd_model = dt_forward_model.DTForwardModel(config)
    self.backwd_model = D3PMBackward(config, self.fwd_model)

  def build_loss_func(self, rng, x0):
    rng, loss_rng = jax.random.split(rng)
    xt, t = self.fwd_model.sample_xt(x0, self.config.hps.num_timesteps, rng)
    loss_fn = functools.partial(self.backwd_model.loss, rng=loss_rng,
                                x0=x0, xt=xt, t=t)
    return loss_fn

  def sample_step(self, params, rng, xt, t):
    model_logits, _ = self.backwd_model.get_logits(params, xt=xt, t=t)

    noise = jax.random.uniform(rng, shape=model_logits.shape)
    nonzero_mask = (t != 0).astype(xt.dtype).reshape(xt.shape[0],
                                                     *([1] * (len(xt.shape))))
    # For numerical precision clip the noise to a minimum value
    noise = jnp.clip(noise, a_min=jnp.finfo(noise.dtype).tiny, a_max=1.)
    gumbel_noise = -jnp.log(-jnp.log(noise))
    sample = jnp.argmax(model_logits + nonzero_mask * gumbel_noise, axis=-1)
    assert sample.shape == xt.shape
    return sample

  def sample_loop(self, state, rng, num_samples=None, conditioner=None):
    rng, prior_rng = jax.random.split(rng)
    if num_samples is None:
      num_samples = self.config.plot_samples // jax.device_count()
    x_start = self.sample_from_prior(
        prior_rng, num_samples, conditioner)
    sampling_gap = self.config.get('sampling_gap', 1)
    def sample_body_fn(step, xt):
      step = step * sampling_gap
      t = jnp.full([xt.shape[0]], self.config.sampling_steps - 1 - step)
      local_rng = jax.random.fold_in(rng, step)
      new_y = self.sample_step(state.ema_params, local_rng, xt, t)
      return new_y

    x0 = jax.lax.fori_loop(0, self.config.sampling_steps // sampling_gap,
                           sample_body_fn, x_start)
    return x0
