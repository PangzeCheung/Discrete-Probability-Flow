#!/bin/bash


data_root="/media/data4/zhangpz/Code/2023/sddm_code_release/data/sddm/synthetic_base_5_d_50/${data_name}"

python -m sddm.synthetic.data.main_datadump \
  --data_root="${data_root}" \
  --data_config="/media/data2/zhangpz/Code/2023/sddm_code_release/sddm/synthetic/data/data_config_base5.py" \
  --num_samples=10000000 \
  --data_config.data_name="${data_name}" \
