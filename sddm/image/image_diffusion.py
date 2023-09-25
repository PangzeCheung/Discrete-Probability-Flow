"""Image diffusion models."""

import os
from flax import jax_utils
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import numpy as np
import PIL
from sddm.common import utils
from MaskGIT.vqgan import VQGAN
import torch


class ImageDiffusion(object):
  """Image model."""

  def __init__(self, config, image_size):
    self.config = config
    self.image_size = image_size

  def log_image(self, images, writer, fname):
    writer.write_images(0, {'data': images[None, ...]})
    if jax.process_index() == 0:
      with open(
          os.path.join(self.config.fig_folder, '%s.png' % fname), 'wb') as f:
        img = PIL.Image.fromarray(images)
        img.save(f)

  def plot_data(self, batch_images, writer):
    images = np.reshape(batch_images,
                        [-1, self.image_size, self.image_size, 3])
    if self.image_size <= 32:
      num_plot = 256
    else:
      num_plot = 64
    images = images[:num_plot].astype(np.uint8)
    images = utils.np_tile_imgs(images)
    self.log_image(images, writer, 'data')


  def encode_batch(self, batch):
    return batch


class VqImageDiffusion(ImageDiffusion):
  """VQ image model."""

  def __init__(self, config, image_size):
    super(VqImageDiffusion, self).__init__(config, image_size)
    model_info = config.vq_model_info
    model = config.vq_model
    '''
    vq_config = config_dict.ConfigDict()
    vq_config.eval_from = config_dict.ConfigDict()
    vq_config.eval_from.xm = model_info[model]['xm']
    vq_config.eval_from.checkpoint_path = model_info[model]['checkpoint_path']
    vq_config.eval_from.step = -1
    '''
    args = {'device':'cuda', 'latent_dim':256, 'image_channels': 3, 'num_codebook_vectors':512, 'beta':0.25}
    torch.set_grad_enabled(False)
    self.VQGAN = VQGAN(args)
    self.VQGAN.load_state_dict(
            torch.load("/media/data4/zhangpz/Code/2023/MaskGIT-pytorch-main/checkpoints_32/vqgan_epoch_435.pt"))
    #self.tokenizer_dict = mm_vq.load_mm_vq_model(vq_config)
    #self.pmap_decode = jax.pmap(self.decode_tokens)
    self.std_rgb = 0.5
    self.mean_rgb = 0.5

  def encode_single_batch(self, batch):
    batch = batch / 255.0
    batch = (batch - self.mean_rgb) / self.std_rgb
    inputs_with_t = jnp.transpose(jnp.squeeze(batch, 0), (0,3,1,2))
    #inputs_with_t = jnp.expand_dims(batch, 1)
    inputs_with_t_torch = torch.from_numpy(np.array(inputs_with_t)).cuda()
    #tokens_with_t = self.tokenizer_dict['tokenizer'](inputs_with_t)
    _, tokens_with_t, _ = self.VQGAN.encode(inputs_with_t_torch)
    tokens_with_t_jnp = jnp.array(tokens_with_t.cpu().detach().numpy())
    tokens = jnp.expand_dims(tokens_with_t_jnp , 0)
   # tokens = jnp.squeeze(tokens_with_t_jnp, 1)
    return tokens

  def decode_tokens(self, tokens):
    tokens_trans = jnp.squeeze(tokens, 0)
    tokens_trans_torch = torch.from_numpy(np.array(tokens_trans)).cuda()
    zq = self.VQGAN.codebook.embedding_process(tokens_trans_torch)
    x0 = self.VQGAN.decode(zq)
    x0_jnp = jnp.array(x0.cpu().detach().numpy())
    x0 = jnp.expand_dims(x0_jnp, 0)
    '''
    tokens_with_t = jnp.expand_dims(tokens, 1)
    #x0 = self.tokenizer_dict['detokenizer'](tokens_with_t)
    zq = self.VQGAN.codebook.embedding_process(tokens_with_t)
    x0 = self.VQGAN.decode(zq)

    x0 = jnp.squeeze(x0, 1)
    '''
    x0 = x0 * self.std_rgb + self.mean_rgb
    x0 = jnp.clip(x0, a_min=0.0, a_max=1.0) * 255

    x0 = jnp.transpose(x0, (0, 1, 3, 4, 2))
    return x0

  def plot_data(self, batch_images, writer):
    ##x0 = self.pmap_decode(batch_images)
    x0 = self.decode_tokens(batch_images)
    x0 = utils.all_gather(x0)
    if jax.process_index() == 0:
      x0 = jax.device_get(jax_utils.unreplicate(x0))
      images = utils.np_tile_imgs(x0.astype(np.uint8))
      self.log_image(images, writer, 'data')

  def plot(self, step, state, rng, sample_fn, writer):
    step_rng_keys = utils.shard_prng_key(rng)
    tokens = sample_fn(state, step_rng_keys)
    print(tokens.shape)
    x0 = self.decode_tokens(tokens)
    x0 = utils.all_gather(x0)
    if jax.process_index() == 0:
      x0 = jax.device_get(jax_utils.unreplicate(x0))
      images = utils.np_tile_imgs(x0.astype(np.uint8))
      self.log_image(images, writer, 'samples_%d' % step)
