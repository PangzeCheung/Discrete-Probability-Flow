import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
import sys
sys.path.insert(0, '../')
from config.eval.piano import get_config as get_eval_config
import lib.utils.bookkeeping as bookkeeping
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import lib.utils.utils as utils
import lib.models.models as models
import lib.models.model_utils as model_utils
import lib.datasets.datasets as datasets
import lib.datasets.dataset_utils as dataset_utils
import lib.sampling.sampling as sampling
import lib.sampling.sampling_utils as sampling_utils


eval_cfg = get_eval_config()
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

dataset = dataset_utils.get_dataset(eval_cfg, device)
data = dataset.data
test_dataset = np.load(eval_cfg.sampler.test_dataset)
condition_dim = eval_cfg.sampler.condition_dim
descramble_key = np.loadtxt(eval_cfg.pianoroll_dataset_path + '/descramble_key.txt')

def descramble(samples):
    return descramble_key[samples.flatten()].reshape(*samples.shape)

descrambled_test_dataset = descramble(test_dataset)

# -------------- Sample the model ------------------
num_samples = 1000
test_data_idx = 1
conditioner = torch.from_numpy(test_dataset[test_data_idx, 0:condition_dim]).view(1, condition_dim)
sampler = sampling_utils.get_sampler(eval_cfg)


x = sampler.sample_x(model, num_samples)
np.save( './x_experiment_music——1000.npy', x.to('cpu').numpy())
x.to('cuda')

'''
x = torch.from_numpy(np.load('./x_experiment_music.npy')).to('cuda')
start_point = 1

for j in range(start_point):

    #x_j = x[j].unsqueeze(0).repeat(num_samples, 1)
    x_j = x[j].unsqueeze(0).repeat(num_samples, 1).to('cuda')
    conditioner = conditioner.repeat(num_samples, 1).to('cuda')
    samples, x_hist, x0_hist = sampler.sample(model, num_samples, 10, conditioner, x = x_j)

    for i in range(num_samples):
        print(samples[i])
        np.save('./piano_1/g_piano'+str(i)+'.npy', samples[i])
        plt.scatter(np.arange(256), samples[i, :], alpha=0.5)
        plt.savefig('./piano_1/g_piano'+str(i)+'.png')
        plt.clf()
'''



#samples, x_hist, x0_hist = descramble(samples), descramble(x_hist), descramble(x0_hist)
