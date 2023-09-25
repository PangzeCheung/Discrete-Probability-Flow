import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
import sys
from config.eval.cifar10 import get_config as get_eval_config
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


num_variance = 10
x = torch.from_numpy(np.load('./test_certainty_xT.npy')).to('cuda')

for i in range(0, 1000):
    x_i = x[i].unsqueeze(0).repeat(num_variance, 1)

    samples, x_hist, x0_hist = sampler.sample(model, num_variance, 10, x = x_i)

    samples = samples.reshape(num_variance, 3, 32, 32)

    res = samples.transpose((0, 2, 3, 1))

    for j in range(10):
        cv2.imwrite(f'res_std_{str(eval_cfg.DPF_type)}/{i:03d}_{j:01d}.png', res[j][:,:,::-1])

