"""Utils."""

import functools
from typing import Any
from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax


@flax.struct.dataclass
class TrainState:
  step: int
  params: Any
  opt_state: Any
  ema_params: Any


def apply_ema(decay, avg, new):
  return jax.tree_map(lambda a, b: decay * a + (1. - decay) * b, avg, new)


def copy_pytree(pytree):
  return jax.tree_map(jnp.array, pytree)


def build_lr_schedule(config):
  """Build lr schedule."""
  if config.lr_schedule == 'constant':
    lr_schedule = lambda step: step * 0 + config.learning_rate
  elif config.lr_schedule == 'updown':
    warmup_steps = int(config.warmup_frac * config.total_train_steps)
    lr_schedule = optax.join_schedules([
        optax.linear_schedule(0, config.learning_rate, warmup_steps),
        optax.linear_schedule(config.learning_rate, 0,
                              config.total_train_steps - warmup_steps)
    ], [warmup_steps])
  elif config.lr_schedule == 'up_exp_down':
    warmup_steps = int(config.warmup_frac * config.total_train_steps)
    lr_schedule = optax.warmup_exponential_decay_schedule(
        init_value=0.0, peak_value=config.learning_rate,
        warmup_steps=warmup_steps, transition_steps=20000,
        decay_rate=0.9, end_value=1e-6
    )
  else:
    raise ValueError('Unknown lr schedule %s' % config.lr_schedule)
  return lr_schedule


def build_optimizer(config):
  """Build optimizer."""
  lr_schedule = build_lr_schedule(config)
  optimizer_name = config.get('optimizer', 'adamw')
  optims = []
  grad_norm = config.get('grad_norm', 0.0)
  if grad_norm > 0.0:
    optims.append(optax.clip_by_global_norm(grad_norm))
  opt_args = {}
  if optimizer_name in ['adamw', 'lamb']:
    opt_args['weight_decay'] = config.get('weight_decay', 0.0)
  optims.append(
      getattr(optax, optimizer_name)(lr_schedule, **opt_args)
  )
  optim = optax.chain(*optims)
  return optim


def init_host_state(params, optimizer):
  state = TrainState(
      step=0,
      params=params,
      opt_state=optimizer.init(params),
      ema_params=copy_pytree(params),
  )
  return jax.device_get(state)


def tf_to_numpy(tf_batch):
  """TF to NumPy, using ._numpy() to avoid copy."""
  # pylint: disable=protected-access
  return jax.tree_map(
      lambda x: x._numpy() if hasattr(x, '_numpy') else x,
      tf_batch)


def numpy_iter(tf_dataset):
  return map(tf_to_numpy, iter(tf_dataset))


def shard_prng_key(prng_key):
  # PRNG keys can used at train time to drive stochastic modules
  # e.g. DropOut. We would like a different PRNG key for each local
  # device so that we end up with different random numbers on each one,
  # hence we split our PRNG key and put the resulting keys into the batch
  return jax.random.split(prng_key, num=jax.local_device_count())


@functools.partial(jax.pmap, axis_name='shard')
def all_gather(x):
  return jax.lax.all_gather(x, 'shard', tiled=True)


def get_per_process_batch_size(batch_size):
  num_devices = jax.device_count()
  assert (batch_size // num_devices * num_devices == batch_size), (
      'Batch size %d must be divisible by num_devices %d', batch_size,
      num_devices)
  batch_size = batch_size // jax.process_count()
  logging.info('Batch size per process: %d', batch_size)
  return batch_size


def categorical_kl_logits(logits1, logits2, eps=1.e-6):
  """KL divergence between categorical distributions.

  Distributions parameterized by logits.

  Args:
    logits1: logits of the first distribution. Last dim is class dim.
    logits2: logits of the second distribution. Last dim is class dim.
    eps: float small number to avoid numerical issues.

  Returns:
    KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
  """
  out = (
      jax.nn.softmax(logits1 + eps, axis=-1) *
      (jax.nn.log_softmax(logits1 + eps, axis=-1) -
       jax.nn.log_softmax(logits2 + eps, axis=-1)))
  return jnp.sum(out, axis=-1)


def meanflat(x):
  """Take the mean over all axes except the first batch dimension."""
  return x.mean(axis=tuple(range(1, len(x.shape))))


def categorical_log_likelihood(x, logits):
  """Log likelihood of a discretized Gaussian specialized for image data.

  Assumes data `x` consists of integers [0, num_classes-1].

  Args:
    x: where to evaluate the distribution. shape = (bs, ...), dtype=int32/int64
    logits: logits, shape = (bs, ..., num_classes)

  Returns:
    log likelihoods
  """
  log_probs = jax.nn.log_softmax(logits)
  x_onehot = jax.nn.one_hot(x, logits.shape[-1])
  return jnp.sum(log_probs * x_onehot, axis=-1)


def log1mexp(x):
  # Computes log(1-exp(-|x|))
  # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
  x = -jnp.abs(x)
  return jnp.where(x > -0.693, jnp.log(-jnp.expm1(x)), jnp.log1p(-jnp.exp(x)))


def binary_hamming_sim(x, y):
  x = jnp.expand_dims(x, axis=1)
  y = jnp.expand_dims(y, axis=0)
  d = jnp.sum(jnp.abs(x - y), axis=-1)
  return x.shape[-1] - d


def binary_exp_hamming_sim(x, y, bd):
  x = jnp.expand_dims(x, axis=1)
  y = jnp.expand_dims(y, axis=0)
  d = jnp.sum(jnp.abs(x - y), axis=-1)
  return jnp.exp(-bd * d)


def binary_mmd(x, y, sim_fn):
  """MMD for binary data."""
  x = x.astype(jnp.float32)
  y = y.astype(jnp.float32)
  kxx = sim_fn(x, x)
  kxx = kxx * (1 - jnp.eye(x.shape[0]))
  kxx = jnp.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

  kyy = sim_fn(y, y)
  kyy = kyy * (1 - jnp.eye(y.shape[0]))
  kyy = jnp.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
  kxy = jnp.sum(sim_fn(x, y))
  kxy = kxy / x.shape[0] / y.shape[0]
  mmd = kxx + kyy - 2 * kxy
  return mmd


def binary_exp_hamming_mmd(x, y, bandwidth=0.1):
  sim_fn = functools.partial(binary_exp_hamming_sim, bd=bandwidth)
  return binary_mmd(x, y, sim_fn)


def binary_hamming_mmd(x, y):
  return binary_mmd(x, y, binary_hamming_sim)


def np_tile_imgs(imgs, *, pad_pixels=1, pad_val=255, num_col=0):
  """NumPy utility: tile a batch of images into a single image.

  Args:
    imgs: np.ndarray: a uint8 array of images of shape [n, h, w, c]
    pad_pixels: int: number of pixels of padding to add around each image
    pad_val: int: padding value
    num_col: int: number of columns in the tiling; defaults to a square

  Returns:
    np.ndarray: one tiled image: a uint8 array of shape [H, W, c]
  """
  if pad_pixels < 0:
    raise ValueError('Expected pad_pixels >= 0')
  if not 0 <= pad_val <= 255:
    raise ValueError('Expected pad_val in [0, 255]')

  imgs = np.asarray(imgs)
  if imgs.dtype != np.uint8:
    raise ValueError('Expected uint8 input')
  # if imgs.ndim == 3:
  #   imgs = imgs[..., None]
  n, h, w, c = imgs.shape
  if c not in [1, 3]:
    raise ValueError('Expected 1 or 3 channels')

  if num_col <= 0:
    # Make a square
    ceil_sqrt_n = int(np.ceil(np.sqrt(float(n))))
    num_row = ceil_sqrt_n
    num_col = ceil_sqrt_n
  else:
    # Make a B/num_per_row x num_per_row grid
    assert n % num_col == 0
    num_row = int(np.ceil(n / num_col))

  imgs = np.pad(
      imgs,
      pad_width=((0, num_row * num_col - n), (pad_pixels, pad_pixels),
                 (pad_pixels, pad_pixels), (0, 0)),
      mode='constant',
      constant_values=pad_val)
  h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
  imgs = imgs.reshape(num_row, num_col, h, w, c)
  imgs = imgs.transpose(0, 2, 1, 3, 4)
  imgs = imgs.reshape(num_row * h, num_col * w, c)

  if pad_pixels > 0:
    imgs = imgs[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
  if c == 1:
    imgs = imgs[..., 0]
  return imgs
