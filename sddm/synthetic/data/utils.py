"""Synthetic data util."""

import matplotlib.pyplot as plt
import numpy as np
from sympy.combinatorics.graycode import GrayCode
from sddm.synthetic.data import toy_data_lib

def plot_samples_variance(samples, out_name, im_size=0, axis=False, im_fmt=None):
  """Plot samples."""
  plt.scatter(samples[:, 0], samples[:, 1], marker='.')
  print(samples[:, 0], samples[:, 1])
  plt.scatter(-1.9124859498697333, 1.5769591892603785, marker='*', color='#FF0000', s=100)
  plt.axis('image')

  if im_size > 0:
    plt.xlim(-im_size, im_size)
    plt.ylim(-im_size, im_size)
  if not axis:
    plt.axis('off')
  if isinstance(out_name, str):
    im_fmt = None
  plt.savefig(out_name, bbox_inches='tight', format=im_fmt)
  plt.close()


def plot_samples(samples, out_name, im_size=0, axis=False, im_fmt=None):
  """Plot samples."""
  plt.scatter(samples[:, 0], samples[:, 1], marker='.')
  plt.axis('image')
  if im_size > 0:
    plt.xlim(-im_size, im_size)
    plt.ylim(-im_size, im_size)
  if not axis:
    plt.axis('off')
  if isinstance(out_name, str):
    im_fmt = None
  plt.savefig(out_name, bbox_inches='tight', format=im_fmt)
  plt.close()


def compress(x, discrete_dim):
  bx = np.binary_repr(int(abs(x)), width=discrete_dim // 2 - 1)
  bx = '0' + bx if x >= 0 else '1' + bx
  return bx

def our_compress(x, discrete_dim, vocab_size):
  bx = np.base_repr(int(abs(x)), base=vocab_size).zfill(discrete_dim // 2)
  return bx


def recover(bx):
  x = int(bx[1:], 2)
  return x if bx[0] == '0' else -x


def our_recover(bx, vocab_size):
  x = int(bx, vocab_size)
  return x


def float2bin(samples, bm, discrete_dim, int_scale):
  bin_list = []
  for i in range(samples.shape[0]):
    x, y = samples[i] * int_scale
    bx, by = compress(x, discrete_dim), compress(y, discrete_dim)
    bx, by = bm[bx], bm[by]
    bin_list.append(np.array(list(bx + by), dtype=int))
  return np.array(bin_list)


def ourfloat2base(samples, discrete_dim, f_scale, int_scale, vocab_size):
  base_list = []
  for i in range(samples.shape[0]):
    x, y = (samples[i] + f_scale) / 2 * int_scale
    bx, by = our_compress(x, discrete_dim, vocab_size), our_compress(y, discrete_dim, vocab_size)
    base_list.append(np.array(list(bx + by), dtype=int))
  return np.array(base_list)



def bin2float(samples, inv_bm, discrete_dim, int_scale):
  """Convert binary to float numpy."""
  floats = []
  for i in range(samples.shape[0]):
    s = ''
    for j in range(samples.shape[1]):
      s += str(samples[i, j])
    x, y = s[:discrete_dim//2], s[discrete_dim//2:]
    x, y = inv_bm[x], inv_bm[y]
    x, y = recover(x), recover(y)
    x /= int_scale
    y /= int_scale
    floats.append((x, y))
  return np.array(floats)


def ourbase2float(samples, discrete_dim, f_scale, int_scale, vocab_size):
  """Convert binary to float numpy."""
  floats = []
  for i in range(samples.shape[0]):
    s = ''
    for j in range(samples.shape[1]):
      s += str(samples[i, j])
    x, y = s[:discrete_dim//2], s[discrete_dim//2:]
    x, y = our_recover(x, vocab_size), our_recover(y, vocab_size)
    x = x / int_scale * 2. - f_scale
    y = y / int_scale * 2. - f_scale
    floats.append((x, y))
  return np.array(floats)


def get_binmap(discrete_dim, binmode):
  """Get binary mapping."""
  b = discrete_dim // 2 - 1
  all_bins = []
  for i in range(1 << b):
    bx = np.binary_repr(i, width=discrete_dim // 2 - 1)
    all_bins.append('0' + bx)
    all_bins.append('1' + bx)
  vals = all_bins[:]
  if binmode == 'gray':
    print('remapping binary repr with gray code')
    a = GrayCode(b)
    vals = []
    for x in a.generate_gray():
      vals.append('0' + x)
      vals.append('1' + x)
  else:
    assert binmode == 'normal'
  bm = {}
  inv_bm = {}
  for i, key in enumerate(all_bins):
    bm[key] = vals[i]
    inv_bm[vals[i]] = key
  return bm, inv_bm


def setup_data(args):
  bm, inv_bm = get_binmap(args.discrete_dim, args.binmode) 
  db = toy_data_lib.OnlineToyDataset(args.data_name)
  args.int_scale = float(db.int_scale)
  args.plot_size = float(db.f_scale)
  return db, bm, inv_bm

def our_setup_data(args):
  db = toy_data_lib.OurPosiOnlineToyDataset(args.data_name, args.vocab_size, args.discrete_dim)
  args.int_scale = float(db.int_scale)
  args.f_scale = float(db.f_scale)
  args.plot_size = float(db.f_scale)
  return db
