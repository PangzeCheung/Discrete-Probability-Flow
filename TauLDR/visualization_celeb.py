import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
import sys
from config.eval.celeb import get_config as get_eval_config
import lib.utils.bookkeeping as bookkeeping
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import lib.utils.utils as utils
import lib.models.models as models
import lib.models.model_utils as model_utils
import lib.sampling.sampling as sampling
import lib.sampling.sampling_utils as sampling_utils
import argparse

import cv2

eval_cfg = get_eval_config()
eval_cfg.device = 'cuda'
train_cfg = bookkeeping.load_ml_collections(Path(eval_cfg.train_config_path))

for item in eval_cfg.train_config_overrides:
    utils.set_in_nested_dict(train_cfg, item[0], item[1])

S = train_cfg.data.S
device = torch.device(eval_cfg.device)

model = model_utils.create_model(train_cfg, device)

loaded_state = torch.load(Path(eval_cfg.checkpoint_path),
    map_location=device)

modified_model_state = utils.remove_module_from_keys(loaded_state['model'])
model.load_state_dict(modified_model_state)

model.eval()

sampler = sampling_utils.get_sampler(eval_cfg)

def imgtrans(x):
    x = np.transpose(x, (1,2,0))
    return x


h = 128

X_T = sampler.sample_x(model, 20)
X_T_save = X_T
X_T_save = X_T_save.cpu().numpy()
np.save('./X_T_inter.npy',X_T_save)

for i in range(10):
    x_a = X_T[i * 2]
    x_b = X_T[i * 2 + 1]

    b, l = x_a.shape
    mask1 = torch.ones((b, l//2)).unsqueeze(1)
    mask2 = torch.zeros((b, l // 2)).unsqueeze(1)
    mask = torch.cat((mask1, mask2), 1).permute(0,2,1)
    mask = mask.reshape(b, -1).to('cuda')
  
    x_m = torch.where(mask>0.1, x_a, x_b)

    x_m_samples, x_hist, x0_hist = sampler.sample(model, 1, 10, x=x_m)
    x_m_samples = x_m_samples.reshape(1, 3, h, h).transpose((0, 2, 3, 1))
    cv2.imwrite('./visualization_celeb_inter/' + str(i) + '_inter' + '.png', x_m_samples[0][:, :, ::-1])

    x_a_samples, x_hist, x0_hist = sampler.sample(model, 1, 10, x=x_a)
    x_a_samples = x_a_samples.reshape(1, 3, h, h).transpose((0, 2, 3, 1))
    cv2.imwrite('./visualization_celeb_inter/' + str(i) + '_image_1' + '.png', x_a_samples[0][:, :, ::-1])

    x_b_samples, x_hist, x0_hist = sampler.sample(model, 1, 10, x=x_b)
    x_b_samples = x_b_samples.reshape(1, 3, h, h).transpose((0, 2, 3, 1))
    cv2.imwrite('./visualization_celeb_inter/' + str(i) + '_image_2' + '.png', x_b_samples[0][:,:,::-1])









