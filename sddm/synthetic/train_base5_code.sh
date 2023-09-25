#!/bin/bash

data_root="data/sddm"
save_root="results/sddm/${config_name?}/${data_name}"

CUDA_VISIBLE_DEVICES=0 python -m sddm.synthetic.train_base_code \
  --data_root="${data_root}" \
  --config="sddm/synthetic/config/${config_name}.py" \
  --config.data_folder="synthetic_base5/${data_name}" \
  --config.save_root="${save_root}" \
  --alsologtostderr \


