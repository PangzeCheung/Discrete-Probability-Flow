"""Forward diffusion models."""

import math
import jax
import jax.numpy as jnp
import numpy as np


class ForwardModel(object):
  """Generic forward model."""

  def __init__(self, num_states):
    self.num_states = num_states

  def rate(self, t):
    raise NotImplementedError

  def rate_mat(self, t):
    raise NotImplementedError

  def transition(self, t):
    raise NotImplementedError

  def transit_between(self, t1, t2):
    raise NotImplementedError

  def sample_xt_with_aux(self, x0, time_duration, rng):
    """Sample x_t and t with aux info returned."""
    bsize = x0.shape[0]
    t_rng, sample_rng = jax.random.split(rng)
    t = jax.random.uniform(t_rng, (bsize,))
    t = t * time_duration
    qt = self.transition(t)
    b = jnp.expand_dims(jnp.arange(bsize), tuple(range(1, x0.ndim)))
    qt0 = qt[b, x0]
    logits = jnp.where(qt0 <= 0.0, -1e9, jnp.log(qt0))
    xt = jax.random.categorical(sample_rng, logits)
    return qt0, xt, t

  def sample_xt(self, x0, time_duration, rng):
    """Sample x_t and t."""
    _, xt, t = self.sample_xt_with_aux(x0, time_duration, rng)
    return xt, t

  def sample_xT(self, x0, time_duration, rng):
    """Sample x_t and t."""
    bsize = x0.shape[0]
    t_rng, sample_rng = jax.random.split(rng)
    t = jnp.ones((bsize,)).astype(jnp.float32)
    t = t * time_duration
    qt = self.transition(t)
    b = jnp.expand_dims(jnp.arange(bsize), tuple(range(1, x0.ndim)))
    qt0 = qt[b, x0]
    logits = jnp.where(qt0 <= 0.0, -1e9, jnp.log(qt0))
    xt = jax.random.categorical(sample_rng, logits)
    return xt

  def sample_from_prior(self, rng, shape):
    raise NotImplementedError


def get_rate_matrix(rate):
  rate = rate - np.diag(np.diag(rate))
  rate = rate - np.diag(np.sum(rate, axis=1))
  eigvals, eigvecs = np.linalg.eigh(rate)
  return (jnp.array(rate, dtype=jnp.float32),
          jnp.array(eigvals, dtype=jnp.float32),
          jnp.array(eigvecs, dtype=jnp.float32))


def get_ot_rate_matrix(num_states, rate_const):
  rate = np.zeros((num_states, num_states))
  for i in range(1, num_states-1):
    rate[i][i - 1] = 1
    rate[i][i] = -2
    rate[i][i + 1] = 1
  rate[0][0] = rate[num_states-1][num_states-1] = -1
  rate[0][1] = rate[num_states-1][num_states-2] = 1
  rate = rate_const * rate
  eigvals, eigvecs = np.linalg.eigh(rate)
  return (jnp.array(rate, dtype=jnp.float32),
          jnp.array(eigvals, dtype=jnp.float32),
          jnp.array(eigvecs, dtype=jnp.float32))


def usvt(eigvecs, inv_eigvecs, diag_embed):
  ns = eigvecs.shape[0]
  u = jnp.reshape(eigvecs, (1, ns, ns))
  vt = jnp.reshape(inv_eigvecs, (1, ns, ns))
  transitions = u @ diag_embed @ vt
  transitions = transitions / jnp.sum(transitions, axis=-1, keepdims=True)
  return transitions


class UniformForward(ForwardModel):
  """Uniform rate."""

  def __init__(self, num_states, rate_const):
    super(UniformForward, self).__init__(num_states=num_states)
    rate = rate_const * np.ones((num_states, num_states))
    self.rate_matrix, self.eigvals, self.eigvecs = get_rate_matrix(rate)

  def rate_mat(self, t):
    return jnp.tile(jnp.expand_dims(self.rate_matrix, axis=0),
                    [t.shape[0], 1, 1])

  def rate(self, y, t):
    del t
    return self.rate_matrix[y]

  def transition(self, t):
    bsize = t.shape[0]
    diag_embed = jax.vmap(jnp.diag)(jnp.exp(
        jnp.reshape(self.eigvals,
                    (1, self.num_states)) * jnp.reshape(t, (bsize, 1))))
    transitions = usvt(self.eigvecs, self.eigvecs.T, diag_embed)
    return transitions

  def transit_between(self, t1, t2):
    return self.transition(t2 - t1)

  def sample_from_prior(self, rng, shape):
    xt = jax.random.randint(rng, shape, minval=0,
                            maxval=self.num_states, dtype=jnp.int32)
    return xt


class UniformOTForward(ForwardModel):
  """Uniform rate."""

  def __init__(self, num_states, rate_const):
    super(UniformOTForward, self).__init__(num_states=num_states)
    self.rate_const = rate_const
    self.rate_matrix, self.eigvals, self.eigvecs = get_ot_rate_matrix(num_states, rate_const)

  def rate_mat(self, t):
    return jnp.tile(jnp.expand_dims(self.rate_matrix, axis=0),
                    [t.shape[0], 1, 1])

  def rate(self, y, t):
    del t
    return self.rate_matrix[y]

  def transition(self, t):
    bsize = t.shape[0]
    diag_embed = jax.vmap(jnp.diag)(jnp.exp(
        jnp.reshape(self.eigvals,
                    (1, self.num_states)) * jnp.reshape(t, (bsize, 1))))
    transitions = usvt(self.eigvecs, self.eigvecs.T, diag_embed)
    return transitions

  def transit_between(self, t1, t2):
    return self.transition(t2 - t1)

  def sample_from_prior(self, rng, shape):
    xt = jax.random.randint(rng, shape, minval=0,
                            maxval=self.num_states, dtype=jnp.int32)
    return xt


  def ot_rate(self, y, t, ll_all):
    voc = ll_all.shape[-1]

    #y: [b, d]
    #ll_all: [b, d, v]
    l_all = jnp.exp(ll_all)

    sel_mask = jax.nn.one_hot(y, voc) #[b,d,v]
    sel_add_mask = jax.nn.one_hot(y + 1, voc)
    sel_sub_mask = jax.nn.one_hot(y - 1, voc)

    ll_sel = sel_mask * l_all
    ll_sel_add = sel_add_mask * l_all
    ll_sel_sub = sel_sub_mask * l_all

    sel_value = jnp.sum(ll_sel, axis=-1, keepdims=True) #[b,d,1]
    sel_add_value = jnp.sum(ll_sel_add, axis=-1, keepdims=True) #[b,d,1]
    sel_sub_value = jnp.sum(ll_sel_sub, axis=-1, keepdims=True) #[b,d,1]

    sel_add_new_value = jnp.where(sel_add_value - sel_value > 0., (sel_add_value - sel_value) / sel_add_value, 0.) #[b,d,1]
    sel_sub_new_value = jnp.where(sel_sub_value - sel_value > 0., (sel_sub_value - sel_value) / sel_sub_value, 0.) #[b,d,1]
    sel_new_value = - sel_add_new_value - sel_sub_new_value #[b,d,1]

    rate = sel_new_value * sel_mask + sel_add_new_value * sel_add_mask + sel_sub_new_value * sel_sub_mask

    return rate


class UniformVariantForward(UniformForward):
  """Variants of uniform."""

  def __init__(self, config):
    super(UniformVariantForward, self).__init__(
        num_states=config.vocab_size, rate_const=config.uniform_rate_const)
    self.t_func = config.t_func

  def _integral(self, t):
    if self.t_func == 'log_sqr':
      return jnp.log(t ** 2 + 1)
    elif self.t_func == 'sqrt_cos':
      return -jnp.sqrt(jnp.cos(jnp.pi / 2 * t))
    else:
      raise ValueError('Unknown t_func %s' % self.t_func)

  def _rate(self, t):
    if self.t_func == 'log_sqr':
      return 2 * t / (t ** 2 + 1)
    elif self.t_func == 'sqrt_cos':
      t = jnp.pi / 2 * t
      tmp = jnp.sin(t) / jnp.sqrt(jnp.cos(t))
      return jnp.pi / 4.0 * tmp
    else:
      raise ValueError('Unknown t_func %s' % self.t_func)

  def rate_mat(self, t):
    rate_scalars = jnp.reshape(self._rate(t), (t.shape[0], 1, 1))
    base = jnp.reshape(self.rate_matrix.T,
                       (1, self.num_states, self.num_states))
    r = base * rate_scalars
    return r

  def rate(self, y, t):
    r = self.rate_mat(t)
    bidx = jnp.expand_dims(jnp.arange(t.shape[0]), tuple(range(1, y.ndim)))
    result = r[bidx, y]
    return result

  def transit_between(self, t1, t2):
    bsize = t2.shape[0]
    d_integral = self._integral(t2) - self._integral(t1)
    diag_embed = jax.vmap(jnp.diag)(jnp.exp(
        jnp.reshape(self.eigvals, (1, self.num_states))
        * jnp.reshape(d_integral, (bsize, 1))))
    transitions = usvt(self.eigvecs, self.eigvecs.T, diag_embed)
    return transitions

  def transition(self, t):
    return self.transit_between(0, t)


def get_fwd_model(config):
  """Get forward model."""
  if config.diffuse_type == 'uniform':
    fwd_model = UniformForward(
        num_states=config.vocab_size, rate_const=config.uniform_rate_const)
  elif config.diffuse_type == 'uniform_variant':
    fwd_model = UniformVariantForward(config)
  elif config.diffuse_type == 'uniform_ot':
    fwd_model = UniformOTForward(num_states=config.vocab_size, rate_const=config.uniform_rate_const)
  else:
    raise ValueError('Unknown diffusion type %s' % config.diffuse_type)
  return fwd_model
