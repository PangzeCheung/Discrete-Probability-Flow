#!/bin/bash

data_root="data/sddm"
save_root="results/sddm/${config_name?}/${data_name}"

CUDA_VISIBLE_DEVICES=0 python -m sddm.synthetic.base_code_test_mmd \
  --data_root="${data_root}" \
  --config="sddm/synthetic/config/base10_code_test_mmd.py" \
  --config.data_folder="synthetic_base10/${data_name}" \
  --config.save_root="${save_root}" \
  --alsologtostderr \





