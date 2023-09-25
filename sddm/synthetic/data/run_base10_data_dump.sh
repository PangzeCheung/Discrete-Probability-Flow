#!/bin/bash


data_root="data/sddm/synthetic_base10/${data_name}"

python -m sddm.synthetic.data.main_datadump \
  --data_root="${data_root}" \
  --data_config="sddm/synthetic/data/base10_code_data_config.py" \
  --num_samples=10000000 \
  --data_config.data_name="${data_name}" \
