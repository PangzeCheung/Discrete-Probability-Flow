#!/bin/bash

data_root="data/sddm"
save_root="results/sddm/${config_name?}/${data_name}"

CUDA_VISIBLE_DEVICES=0 python -m sddm.synthetic.binary_graycode_test_mmd \
  --data_root="${data_root}" \
  --config="sddm/synthetic/config/binary_graycode_test_mmd.py" \
  --config.data_folder="synthetic_binary/${data_name}" \
  --config.save_root="${save_root}" \
  --alsologtostderr \













