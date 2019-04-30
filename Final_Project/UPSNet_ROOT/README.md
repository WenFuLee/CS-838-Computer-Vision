# UPSNet: A Unified Panoptic Segmentation Network

# Introduction
UPSNet is initially described in a [CVPR 2019 oral](https://arxiv.org/abs/1901.03784) paper.




# Disclaimer

This repository is tested under Python 3.6, PyTorch 0.4.1. And model training is done with 16 GPUs by using [horovod](https://github.com/horovod/horovod). It should also work under Python 2.7 / PyTorch 1.0 and with 4 GPUs.

# License
© Uber, 2018-2019. Licensed under the Uber Non-Commercial License.

# Citing UPSNet

If you find UPSNet is useful in your research, please consider citing:
```
@inproceedings{xiong19upsnet,
    Author = {Yuwen Xiong, Renjie Liao, Hengshuang Zhao, Rui Hu, Min Bai, Ersin Yumer, Raquel Urtasun},
    Title = {UPSNet: A Unified Panoptic Segmentation Network},
    Conference = {CVPR},
    Year = {2019}
}
```


# Main Results

Cityscapes

|                | PQ   | SQ   | RQ   | PQ<sup>Th</sup> | PQ<sup>St</sup> |
|----------------|------|------|------|-----------------|-----------------|
| UPSNet-50      | 59.3 | 79.7 | 73.0 | 54.6            | 62.7            |

# Requirements: Software

We recommend using Anaconda3 as it already includes many common packages.


# Requirements: Hardware

We recommend using 4~16 GPUs with at least 11 GB memory to train our model.

# Installation

Clone this repo to `$UPSNet_ROOT`

Run `init.sh` to build essential C++/CUDA modules and download pretrained model.

For Cityscapes:

Assuming you already downloaded Cityscapes dataset at `$CITYSCAPES_ROOT` and TrainIds label images are generated, please create a soft link by `ln -s $CITYSCAPES_ROOT data/cityscapes` under `UPSNet_ROOT`, and run `init_cityscapes.sh` to prepare Cityscapes dataset for UPSNet.

Training:

`python upsnet/upsnet_end2end_train.py --cfg upsnet/experiments/upsnet_resnet50_cityscapes_1gpu.yaml`

Test:

`python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_resnet50_cityscapes_1gpu.yaml`

Attack and Defend:

`python upsnet/upsnet_end2end_test_attack.py --cfg upsnet/experiments/upsnet_resnet50_cityscapes_1gpu.yaml`
- Attack method: class PGDAttack(object)
- Defend method: def applyBiFilter(advIn, oriIn)

We provide serveral config files (1/4/16 GPUs for Cityscapes dataset) under upsnet/experiments folder.
