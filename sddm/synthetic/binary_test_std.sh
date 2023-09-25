#!/bin/bash

data_root="/media/data2/zhangpz/Code/2023/sddm_code_release/data/sddm"
save_root="/media/data4/zhangpz/Code/2023/sddm_code_release/results/sddm/ebm_binary_ot_lbjf_config_testwoot/${data_name}"

CUDA_VISIBLE_DEVICES=5 python -m sddm.synthetic.binary_graycode_test_std \
  --data_root="${data_root}" \
  --config="/media/data2/zhangpz/Code/2023/sddm/sddm/synthetic/config/binary_graycode_test_std.py" \
  --config.data_folder="/media/data2/zhangpz/Code/2023/sddm_code_release/data/sddm/synthetic/${data_name}" \
  --config.save_root="${save_root}" \
  --alsologtostderr \





