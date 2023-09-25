# Discrete-Probability-Flow

The source code for our paper "Formulating Discrete Probability Flow Through Optimal Transport", Pengze Zhang*, Hubery Yin*, Chen Li, Xiaohua Xie, NeurIPS 2023.

<img width="1059" alt="DPF" src="https://github.com/PangzeCheung/Discrete-Probability-Flow/assets/37894893/7acf5a4b-98b1-4c0d-8ca9-6d6f73af8a65">


## Abstract

Continuous diffusion models are commonly acknowledged to display a deterministic probability flow, whereas discrete diffusion models do not. In this paper, we aim to establish the fundamental theory for the probability flow of discrete diffusion models. Specifically, we first prove that the continuous probability flow is the Monge optimal transport map under certain mild conditions, and also present a equivalent evidence for discrete cases.  In view of these findings, we are then able to define the discrete probability flow in line with the principles of optimal transport. Finally, drawing upon our newly established definitions, we propose a novel sampling method that surpasses previous discrete diffusion models in its ability to generate more certain outcomes.

## Discrete Probability Flow on SDDM (Toy dataset)

### 1) Get start

* Python 3.9.0
* jax 0.4.8        
* jaxlib 0.4.7
* CUDA 12.1
* NVIDIA A100 40GB PCIe

Open the directory for sddm.
``` bash
cd sddm
```

### Generate your synthetic dataset

The synthetic data can be divided into 7 categories: 2spirals, 8gaussians, checkerboard, circles, moons, pinwheel, swissroll. You can set 'data_name' for selection.

**Binary graycode**
``` bash
data_name=XXX bash sddm/synthetic/data/run_binary_data_dump.sh
```

**Base5 code**
``` bash
data_name=XXX bash sddm/synthetic/data/run_base5_data_dump.sh
```

**Base10 code**
``` bash
data_name=XXX bash sddm/synthetic/data/run_base10_data_dump.sh
```

You can also directly download our **[synthetic dataset](https://drive.google.com/drive/folders/1Y-1D6ICI7hMQjjivl_dsFkktC2jz1phn?usp=drive_link)** into ./sddm/data

### Train on the synthetic dataset

**Binary graycode**
``` bash
data_name=XXX config_name=binary_graycode bash sddm/synthetic/train_binary_graycode.sh
```
**Base5 code**
``` bash
data_name=XXX config_name=base5_code bash ./sddm/synthetic/train_base5_code.sh
```
**Base10 code**
``` bash
data_name=XXX config_name=base10_code bash ./sddm/synthetic/train_base10_code.sh
```

You can also directly download our **[pre-trained model](https://drive.google.com/drive/folders/1Yp3Lh1HQVQvTxPlQQe4Dw5WYPkicbdwg?usp=drive_link)** into ./sddm/results

### Test MMD
Please switch 'sampler_type' in 'sddm/synthetic/config/*.py' to choose lbjf or dpf sampling.

**Binary graycode**
``` bash
data_name=XXX config_name=binary_graycode bash sddm/synthetic/binary_test_mmd.sh 
```

**Base5 code**
``` bash
data_name=XXX config_name=base5_code bash sddm/synthetic/base5_test_mmd.sh
```

**Base10 code**
``` bash
data_name=XXX config_name=base10_code bash sddm/synthetic/base10_test_mmd.sh
```

### Test CSV
Please switch 'sampler_type' in 'sddm/synthetic/config/*.py' to choose lbjf or dpf sampling.

**Binary graycode**
``` bash
data_name=XXX config_name=binary_graycode bash sddm/synthetic/binary_test_std.sh 
```

**Base5 code**
``` bash
data_name=XXX config_name=base5_code bash sddm/synthetic/base5_test_std.sh
```

**Base10 code**
``` bash
data_name=XXX config_name=base10_code bash sddm/synthetic/base10_test_std.sh
```

## Discrete Probability Flow on TauLDR (Cifar10)

### 1) Get start

* Python 3.9.7
* pytorch 1.12.1        
* torchvision 0.13.1
* CUDA 11.3
* NVIDIA A100 40GB PCIe

Open the directory for TauLDR.
``` bash
cd TauLDR
```

### 2) Prepare the pre-trained model

The model is provided by **[TauLDR](https://github.com/andrew-cr/tauLDR)**. You can directly download the **[model](https://www.dropbox.com/scl/fo/zmwsav82kgqtc0tzgpj3l/h?dl=0&rlkey=k6d2bp73k4ifavcg9ldjhgu0s)** into ./TauLDR/models/cifar10

### 3) Generate samples for evaluation
Please download the **[x_T](https://drive.google.com/file/d/1FBddRPNh2Z_PIv_2m8YgxAmI1hnqf_TX/view?usp=drive_link)** into ./TauLDR for reproduction.

Switch 'DPF_type' in 'TauLDR/config/eval/cifar10.py' to 0 / 1 to choose TauLDR / DPF sampling.

``` bash
python generate_test_certainty_data.py
```

### 4) Test CSD

``` bash
python test_std.py --DPF_type X
```

### 5) Test class-std
Please download the pretrained **[Cifar10 classifier](https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/master/pretrained_models/resnet20-12fca82f.th)**, and put it in ./TauLDR

``` bash
python test_class_std.py --DPF_type X
```

### 6) Test class-entropy

``` bash
python test_entropy.py --DPF_type X
```

### 7) Visualization
Please download the **[x_T](https://drive.google.com/file/d/15q2rhL3qfY2nj1qqEgDQ3sKgkbU_2suT/view?usp=drive_link)** into ./TauLDR for reproduction our Figure 12.

Switch 'DPF_type' in 'TauLDR/config/eval/cifar10.py' to 0 / 1 to choose TauLDR / DPF sampling.

``` bash
python visualization.py
```

## Citation

```tex
@inproceedings{
zhang2023formulating,
title={Formulating Discrete Probability Flow Through Optimal Transport},
author={Pengze Zhang and Hubery Yin and Chen Li and Xiaohua Xie},
booktitle={Advances in Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=I9GNrInbdf}
}
```

## Acknowledgement 

We build our project based on **[SDDM](https://openreview.net/forum?id=BYWWwSY2G5s)** and **[TauLDR](https://github.com/andrew-cr/tauLDR)**. We thank them for their wonderful work and code release.
