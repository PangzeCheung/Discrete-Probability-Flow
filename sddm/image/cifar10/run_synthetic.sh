#!/bin/bash

data_name="cifar10"
data_root="/media/data2/zhangpz/Code/2023/sddm_code_release/data/sddm"
save_root="/media/data2/zhangpz/Code/2023/sddm_code_release/results/sddm/${config_name?}/${data_name}"

CUDA_VISIBLE_DEVICES=1 python -m sddm.image.cifar10.main_cifar10 \
  --data_root="${data_root}" \
  --config="/media/data2/zhangpz/Code/2023/sddm_code_release/sddm/image/cifar10/config/${config_name}.py" \
  --config.data_folder="/media/data2/zhangpz/Code/2023/sddm_code_release/data/sddm/synthetic/${data_name}" \
  --config.save_root="${save_root}" \
  --alsologtostderr \
