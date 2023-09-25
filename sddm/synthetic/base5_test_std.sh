#!/bin/bash

data_root="/media/data2/zhangpz/Code/2023/sddm_code_release/data/sddm"
save_root="/media/data2/zhangpz/Code/2023/sddm_code_release/results/sddm/ebm_cat_ot_config_base5/${data_name}"

CUDA_VISIBLE_DEVICES=1 python -m sddm.synthetic.base_code_test_std \
  --data_root="${data_root}" \
  --config="/media/data2/zhangpz/Code/2023/sddm/sddm/synthetic/config/base5_code_test_std.py" \
  --config.data_folder="/media/data2/zhangpz/Code/2023/sddm_code_release/data/sddm/synthetic_base5/${data_name}" \
  --config.save_root="${save_root}" \
  --alsologtostderr \





