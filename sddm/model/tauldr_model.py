"""Tau ldr."""

import jax
import jax.numpy as jnp

from sddm.model import backward_model
from sddm.model import forward_model


class TauLDRBackward(backward_model.BackwardModel):
  """Tau LDR backward model, from https://github.com/andrew-cr/tauLDR"""

  def __init__(self, config):
    super(TauLDRBackward, self).__init__(config)
    self.fwd_model = forward_model.get_fwd_model(self.config)
    self.net = backward_model.FreeformTransformer(config)

  def _sample_categorical(self, rng, prob):
    rng, local_rng = jax.random.split(rng)
    lob_prob = jnp.where(prob <= 0, -1e9, jnp.log(prob))
    val = jax.random.categorical(local_rng, lob_prob)
    return rng, val

  def get_ratio(self, params, xt, t, xt_target=None):
    raise NotImplementedError

  def get_logits(self, params, xt, t):
    pass

  def get_logprob(self, params, xt, t, xt_target=None):
    del xt_target
    b = jnp.expand_dims(jnp.arange(xt.shape[0]), tuple(range(1, xt.ndim)))
    x0_logits = self.net.apply({'params': params}, x=xt, t=t)
    p0t = jax.nn.softmax(x0_logits, axis=-1)
    qt0 = jnp.clip(self.fwd_model.transition(t), a_min=1e-8)
    qt0_denorm = jnp.transpose(qt0, (0, 2, 1))[b, xt]

    qt0_numer = jnp.expand_dims(qt0, axis=list(range(1, xt.ndim - 1)))
    inner_sum = (p0t / qt0_denorm) @ qt0_numer
    xt_onehot = jax.nn.one_hot(xt, self.config.vocab_size)
    ll_all = inner_sum * (1 - xt_onehot) + xt_onehot
    ll_all = jnp.where(ll_all < 1e-35, -1e9, jnp.log(ll_all))
    ll_xt = jnp.zeros(ll_all.shape[:-1])
    return ll_all, ll_xt

  def loss(self, params, rng, x0, xt, t):
    eps = 1e-9
    config = self.config
    qt0 = self.fwd_model.transition(t)
    qt0 = jnp.clip(qt0, a_min=1e-8)
    rate_mat = self.fwd_model.rate_mat(t)
    bsize = xt.shape[0]
    xt = jnp.reshape(xt, [bsize, -1])
    d = xt.shape[1]
    s = config.vocab_size
    xt_onehot = jax.nn.one_hot(xt, s)
    cat_dims = xt.shape[1]
    b = jnp.expand_dims(jnp.arange(bsize), axis=1)

    rate_given_xt = rate_mat[b, xt]
    rate_given_xt = rate_given_xt * (1 - xt_onehot)
    rate_xt_offdiag = jnp.sum(rate_given_xt, axis=-1)
    rng, dimcat = self._sample_categorical(rng, rate_xt_offdiag)

    rate_newval = rate_given_xt[jnp.arange(bsize), dimcat]
    rng, valcat = self._sample_categorical(rng, rate_newval)
    dimcat_onehot = jax.nn.one_hot(dimcat, cat_dims, dtype=jnp.int32)
    valcat = jnp.expand_dims(valcat, axis=-1)
    xtilde = xt * (1 - dimcat_onehot) + dimcat_onehot * valcat

    if config.tauldr_onepass:
      x_logits = self.net.apply({'params': params}, x=xtilde, t=t)
      reg_x = xtilde
    else:
      x_logits = self.net.apply({'params': params}, x=xt, t=t)
      reg_x = xt
    p0t_reg = jax.nn.softmax(x_logits, axis=2)

    # first term
    reg_x_onehot = jax.nn.one_hot(reg_x, s)
    rate2xt = jnp.transpose(rate_mat, (0, 2, 1))[b, reg_x]
    rate2xt = rate2xt * (1 - reg_x_onehot)
    reg_tmp = rate2xt @ jnp.transpose(qt0, (0, 2, 1))
    qt0_denom_reg = jnp.transpose(qt0, (0, 2, 1))[b, reg_x]
    reg_term = jnp.sum((p0t_reg / qt0_denom_reg) * reg_tmp, axis=(1, 2))

    # second term

    if config.tauldr_onepass:
      p0t_sig = p0t_reg
    else:
      x_logits = self.net.apply({'params': params}, x=xtilde, t=t)
      p0t_sig = jax.nn.softmax(x_logits, axis=2)

    outer_qt0_numer_sig = qt0[
        jnp.repeat(jnp.arange(bsize), d * s),
        jnp.repeat(jnp.ravel(x0), s),
        jnp.tile(jnp.arange(s), [bsize * d])
    ]
    outer_qt0_numer_sig = jnp.reshape(outer_qt0_numer_sig,
                                      [bsize, d, s])

    outer_qt0_denom_sig = qt0[
        jnp.repeat(jnp.arange(bsize), d),
        jnp.ravel(x0),
        jnp.ravel(xtilde)
    ] + eps

    qt0_denom_sig = jnp.transpose(qt0, (0, 2, 1))[b, xtilde] + eps

    inner_log_sig = jnp.log(
        (p0t_sig / qt0_denom_sig) @ qt0 + eps
    )
    xtilde_onehot = jax.nn.one_hot(xtilde, s)
    outer_rate_sig = rate_mat[
        jnp.repeat(jnp.arange(bsize), d * s),
        jnp.tile(jnp.arange(s), [bsize * d]),
        jnp.repeat(jnp.ravel(xtilde), s),
    ]
    outer_rate_sig = jnp.reshape(outer_rate_sig, [bsize, d, s])

    oss_tmp = outer_qt0_numer_sig / jnp.reshape(outer_qt0_denom_sig,
                                                [bsize, d, 1])
    outer_sum_sig = jnp.sum(
        (1 - xtilde_onehot) * outer_rate_sig * oss_tmp * inner_log_sig,
        axis=(1, 2)
    )

    rate_row_sums = -rate_mat[
        jnp.repeat(jnp.arange(bsize), s),
        jnp.tile(jnp.arange(s), [bsize]),
        jnp.tile(jnp.arange(s), [bsize]),
    ]
    rate_row_sums = jnp.reshape(rate_row_sums, [bsize, s])

    base_z_tmp = rate_row_sums[
        jnp.repeat(jnp.arange(bsize), d),
        jnp.ravel(xtilde)
    ]
    base_z_tmp = jnp.reshape(base_z_tmp, [bsize, d])
    base_z = jnp.sum(base_z_tmp, axis=1)

    z_subtraction = base_z_tmp
    z_addition = rate_row_sums
    z_sig_norm = (jnp.reshape(base_z, (bsize, 1, 1)) -
                  jnp.reshape(z_subtraction, (bsize, d, 1)) +
                  jnp.reshape(z_addition, (bsize, 1, s)))

    rate_sig_norm = rate_mat[
        jnp.repeat(jnp.arange(bsize), d * s),
        jnp.tile(jnp.arange(s), [bsize * d]),
        jnp.repeat(jnp.ravel(xtilde), s),
    ]
    rate_sig_norm = jnp.reshape(rate_sig_norm, [bsize, d, s])
    qt0_sig_norm_numer = qt0[
        jnp.repeat(jnp.arange(bsize), d * s),
        jnp.repeat(jnp.ravel(x0), s),
        jnp.tile(jnp.arange(s), [bsize * d])
    ]
    qt0_sig_norm_numer = jnp.reshape(qt0_sig_norm_numer, [bsize, d, s])
    qt0_sig_norm_denom = qt0[
        jnp.repeat(jnp.arange(bsize), d),
        jnp.ravel(x0),
        jnp.ravel(xtilde)
    ] + eps
    qt0_sig_norm_denom = jnp.reshape(qt0_sig_norm_denom, [bsize, d])

    sig_norm_numer = rate_sig_norm * qt0_sig_norm_numer * (1 - xtilde_onehot)
    sig_norm_denom = z_sig_norm * jnp.expand_dims(qt0_sig_norm_denom, -1) + eps
    sig_norm = jnp.sum(sig_norm_numer / sig_norm_denom, axis=(1, 2))

    sig_mean = jnp.mean(-outer_sum_sig / sig_norm)
    reg_mean = jnp.mean(reg_term)
    neg_elbo = sig_mean + reg_mean
    aux = {'loss': neg_elbo}
    return neg_elbo, aux
