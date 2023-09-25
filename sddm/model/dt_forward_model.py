"""Discrete time Forward diffusion models, largely based on D3PM code."""

import jax
import jax.numpy as jnp
import numpy as onp
import scipy


def get_diffusion_betas(spec, num_timesteps):
  """Get betas from the hyperparameters."""
  if spec.type == 'linear':
    # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
    # To be used with Gaussian diffusion models in continuous and discrete
    # state spaces.
    # To be used with transition_mat_type = 'gaussian'
    return onp.linspace(spec.start, spec.stop, num_timesteps)
  elif spec.type == 'cosine':
    # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
    # To be used with transition_mat_type = 'uniform'.
    steps = (
        onp.arange(num_timesteps + 1, dtype=onp.float64) /
        num_timesteps)
    alpha_bar = onp.cos((steps + 0.008) / 1.008 * onp.pi / 2)
    betas = onp.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
    return betas
  elif spec.type == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
    # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
    # To be used with absorbing state models.
    # ensures that the probability of decaying to the absorbing state
    # increases linearly over time, and is 1 for t = T-1 (the final time).
    # To be used with transition_mat_type = 'absorbing'
    return 1. / onp.linspace(num_timesteps, 1., num_timesteps)
  else:
    raise NotImplementedError(spec.type)


class DTForwardModel(object):
  """Generic discrete time forward model."""

  def _get_full_transition_mat(self, t):
    """Computes transition matrix for q(x_t|x_{t-1}).

    Contrary to the band diagonal version, this method constructs a transition
    matrix with uniform probability to all other states.

    Args:
      t: timestep. integer scalar.

    Returns:
      Q_t: transition matrix. shape = (vocab_size, vocab_size).
    """
    beta_t = self.betas[t]
    mat = onp.full(shape=(self.num_states, self.num_states),
                   fill_value=beta_t / float(self.num_states),
                   dtype=onp.float64)
    diag_indices = onp.diag_indices_from(mat)
    diag_val = 1. - beta_t * (self.num_states - 1.) / self.num_states
    mat[diag_indices] = diag_val
    return mat

  def _get_transition_mat(self, t):
    r"""Computes transition matrix for q(x_t|x_{t-1}).

    This method constructs a transition
    matrix Q with
    Q_{ij} = beta_t / num_pixel_vals       if |i-j| <= self.transition_bands
             1 - \sum_{l \neq i} Q_{il} if i==j.
             0                          else.

    Args:
      t: timestep. integer scalar (or numpy array?)

    Returns:
      Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    if self.transition_bands is None:
      return self._get_full_transition_mat(t)
    # Assumes num_off_diags < num_pixel_vals
    beta_t = self.betas[t]

    mat = onp.zeros((self.num_states, self.num_states),
                    dtype=onp.float64)
    off_diag = onp.full(shape=(self.num_states - 1,),
                        fill_value=beta_t/float(self.num_states),
                        dtype=onp.float64)
    for k in range(1, self.transition_bands + 1):
      mat += onp.diag(off_diag, k=k)
      mat += onp.diag(off_diag, k=-k)
      off_diag = off_diag[:-1]

    # Add diagonal values such that rows sum to one.
    diag = 1. - mat.sum(1)
    mat += onp.diag(diag, k=0)
    return mat

  def _get_absorbing_transition_mat(self, t):
    """Computes transition matrix for q(x_t|x_{t-1}).

    Has an absorbing state for pixelvalues vocab_size//2.

    Args:
      t: timestep. integer scalar.

    Returns:
      Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    beta_t = self.betas[t]

    diag = onp.full(shape=(self.num_states,), fill_value=1. - beta_t,
                    dtype=onp.float64)
    mat = onp.diag(diag, k=0)
    # Add beta_t to the num_pixel_vals/2-th column for the absorbing state.
    mat[:, self.num_states // 2] += beta_t

    return mat

  def _get_gaussian_transition_mat(self, t):
    r"""Computes transition matrix for q(x_t|x_{t-1}).

    This method constructs a transition matrix Q with
    decaying entries as a function of how far off diagonal the entry is.
    Normalization option 1:
    Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
             1 - \sum_{l \neq i} Q_{il}  if i==j.
             0                          else.

    Normalization option 2:
    tilde{Q}_{ij} =  softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                     0                        else.

    Q_{ij} =  tilde{Q}_{ij} / sum_l{tilde{Q}_{lj}}

    Args:
      t: timestep. integer scalar (or numpy array?)

    Returns:
      Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    if self.transition_bands:
      transition_bands = self.transition_bands
    else:
      transition_bands = self.num_states - 1

    beta_t = self.betas[t]

    mat = onp.zeros((self.num_states, self.num_states),
                    dtype=onp.float64)

    # Make the values correspond to a similar type of gaussian as in the
    # gaussian diffusion case for continuous state spaces.
    values = onp.linspace(start=0., stop=255., num=self.num_states,
                          endpoint=True, dtype=onp.float64)
    values = values * 2./ (self.num_states - 1.)
    values = values[:transition_bands+1]
    values = -values * values / beta_t

    values = onp.concatenate([values[:0:-1], values], axis=0)
    values = scipy.special.softmax(values, axis=0)
    values = values[transition_bands:]
    for k in range(1, transition_bands + 1):
      off_diag = onp.full(shape=(self.num_states - k,),
                          fill_value=values[k],
                          dtype=onp.float64)

      mat += onp.diag(off_diag, k=k)
      mat += onp.diag(off_diag, k=-k)

    # Add diagonal values such that rows and columns sum to one.
    # Technically only the ROWS need to sum to one
    # NOTE: this normalization leads to a doubly stochastic matrix,
    # which is necessary if we want to have a uniform stationary distribution.
    diag = 1. - mat.sum(1)
    mat += onp.diag(diag, k=0)

    return mat

  def __init__(self, config):
    self.num_states = config.vocab_size
    self.eps = 1.e-6
    self.transition_bands = config.hps.transition_bands
    self.num_timesteps = config.hps.num_timesteps
    self.betas = get_diffusion_betas(config.hps.diffusion_betas,
                                     config.hps.num_timesteps)
    self.transition_mat_type = config.hps.transition_mat_type

    if self.transition_mat_type == 'uniform':
      q_one_step_mats = [self._get_transition_mat(t)
                         for t in range(0, self.num_timesteps)]
    elif self.transition_mat_type == 'gaussian':
      q_one_step_mats = [self._get_gaussian_transition_mat(t)
                         for t in range(0, self.num_timesteps)]
    elif self.transition_mat_type == 'absorbing':
      q_one_step_mats = [self._get_absorbing_transition_mat(t)
                         for t in range(0, self.num_timesteps)]
    else:
      raise ValueError(
          f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' "
          f", but is {self.transition_mat_type}"
          )
    self.q_onestep_mats = onp.stack(q_one_step_mats, axis=0)
    assert self.q_onestep_mats.shape == (self.num_timesteps,
                                         self.num_states,
                                         self.num_states)
    # Construct transition matrices for q(x_t|x_start)
    q_mat_t = self.q_onestep_mats[0]
    q_mats = [q_mat_t]
    for t in range(1, self.num_timesteps):
      # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
      q_mat_t = onp.tensordot(q_mat_t, self.q_onestep_mats[t],
                              axes=[[1], [0]])
      q_mats.append(q_mat_t)
    self.q_mats = onp.stack(q_mats, axis=0)
    assert self.q_mats.shape == (self.num_timesteps, self.num_states,
                                 self.num_states), self.q_mats.shape
    # Don't precompute transition matrices for q(x_{t-1} | x_t, x_start)
    # Can be computed from self.q_mats and self.q_one_step_mats.
    # Only need transpose of q_onestep_mats for posterior computation.
    self.transpose_q_onestep_mats = onp.transpose(self.q_onestep_mats,
                                                  axes=(0, 2, 1))
    self.transpose_q_onestep_mats = jnp.asarray(self.transpose_q_onestep_mats,
                                                dtype=jnp.float32)
    self.q_mats = jnp.asarray(self.q_mats, dtype=jnp.float32)

  def _at(self, a, t, x):
    """Extract coefficients at specified timesteps t and conditioning data x.

    Args:
      a: np.ndarray: plain NumPy float64 array of constants indexed by time.
      t: jnp.ndarray: Jax array of time indices, shape = (batch_size,).
      x: jnp.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
        (Noisy) data. Should not be of one hot representation, but have integer
        values representing the class values.

    Returns:
      a[t, x]: jnp.ndarray: Jax array.
    """
    # a = jnp.asarray(a, dtype=jnp.float32)
    t_broadcast = jnp.expand_dims(t, tuple(range(1, x.ndim)))
    # x.shape = (bs, height, width, channels)
    # t_broadcast_shape = (bs, 1, 1, 1)
    # a.shape = (num_timesteps, num_pixel_vals, num_pixel_vals)
    # out.shape = (bs, height, width, channels, num_pixel_vals)
    # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
    return a[t_broadcast, x]

  def at(self, t, x):
    return self._at(self.q_mats, t, x)

  def _at_onehot(self, a, t, x):
    """Extract coefficients at specified timesteps t and conditioning data x.

    Args:
      a: np.ndarray: plain NumPy float64 array of constants indexed by time.
      t: jnp.ndarray: Jax array of time indices, shape = (bs,).
      x: jnp.ndarray: jax array, shape (bs, ..., num_pixel_vals), float32 type.
        (Noisy) data. Should be of one-hot-type representation.

    Returns:
      out: jnp.ndarray: Jax array. output of dot(x, a[t], axis=[[-1], [1]]).
        shape = (bs, ..., num_pixel_vals)
    """
    # a = jnp.asarray(a, dtype=jnp.float32)

    # x.shape = (bs, dim, num_pixel_vals)
    # a[t]shape = (bs, num_pixel_vals, num_pixel_vals)
    # out.shape = (bs, dim, num_pixel_vals)
    return jnp.matmul(x, a[t], precision=jax.lax.Precision.HIGHEST)

  def at_onehot(self, t, x):
    return self._at_onehot(self.q_mats, t, x)

  def at_onestep(self, t, x_t):
    return self._at(self.q_onestep_mats, t, x_t)

  def at_transpose_onestep(self, t, x):
    return self._at(self.transpose_q_onestep_mats, t, x)

  def sample_from_prior(self, rng, shape):
    xt = jax.random.randint(rng, shape, minval=0,
                            maxval=self.num_states, dtype=jnp.int32)
    return xt

  def q_sample(self, q_prob, noise):
    """Sample from q(x_t | x_start) (i.e. add noise to the data)."""
    assert noise.shape == q_prob.shape
    logits = jnp.log(q_prob + self.eps)

    # To avoid numerical issues clip the noise to a minimum value
    noise = jnp.clip(noise, a_min=jnp.finfo(noise.dtype).tiny, a_max=1.)
    gumbel_noise = - jnp.log(-jnp.log(noise))
    return jnp.argmax(logits + gumbel_noise, axis=-1)

  def sample_xt_with_aux(self, x0, num_timesteps, rng):
    """Sample x_t and t with aux info returned."""
    bsize = x0.shape[0]
    noise_rng, time_rng = jax.random.split(rng)
    noise = jax.random.uniform(noise_rng,
                               shape=x0.shape + (self.num_states,))
    t = jax.random.randint(time_rng, shape=(bsize,), minval=0,
                           maxval=num_timesteps, dtype=jnp.int32)
    q_prob = self.at(t, x0)
    x_t = self.q_sample(q_prob=q_prob, noise=noise)
    return q_prob, x_t, t

  def sample_xt(self, x0, num_timesteps, rng):
    """Sample x_t and t."""
    _, xt, t = self.sample_xt_with_aux(x0, num_timesteps, rng)
    return xt, t

