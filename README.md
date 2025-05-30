## Introduction

This is the official repository for our **ISBI 2025** paper:

> **Open-Set Semi-Supervised Learning for Long-Tailed Medical Datasets**</br>
> Daniya Najiha, Jean Lahoud, Mustansar Fiaz, Amandeep Kumar, Hisham Cholakkal</br>

[[`Paper`](https://arxiv.org/abs/2505.14846)]  [[`Models and Logs`](https://drive.google.com/drive/folders/1pLU6tqxMls55CBRvCgZmDBfHLXm7jGMv?usp=sharing)]]

## Preparation

### Required Packages

We suggest first creating a conda environment:

```sh
conda create --name openltr python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

### Datasets

Please put the datasets in the ``./data`` folder (or create soft links) as follows:
```
OpenLTR
├── config
    └── ...
├── data
    ├── cifar10
        └── cifar-10-batches-py
    └── cifar100
        └── cifar-100-python
    └── imagenet30
        └── filelist
        └── one_class_test
        └── one_class_train
    └── ood_data
├── semilearn
    └── ...
└── ...  
```

The data of ImageNet-30 can be downloaded in [one_class_train](https://drive.google.com/file/d/1B5c39Fc3haOPzlehzmpTLz6xLtGyKEy4/view) and [one_class_test](https://drive.google.com/file/d/13xzVuQMEhSnBRZr-YaaO08coLU2dxAUq/view).

The out-of-dataset testing data for extended open-set evaluation can be downloaded in [this link](https://drive.google.com/drive/folders/1IjDLYfpfsMVuzf_NmqQPoHDH0KAd94gn?usp=sharing).

## Usage

We implement [IOMatch](./semilearn/algorithms/iomatch/iomatch.py) using the codebase of [USB](https://github.com/microsoft/Semi-supervised-learning).

### Training

Here is an example to train IOMatch on CIFAR-100 with the seen/unseen split of "50/50" and 25 labels per seen class (*i.e.*, the task <u>CIFAR-50-1250</u> with 1250 labeled samples in total). 

```sh
# seed = 1
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_cifar100_1250_1.yaml
```

Training IOMatch on other datasets with different OSSL settings can be specified by a config file:
```sh
# CIFAR10, seen/unseen split of 6/4, 25 labels per seen class (CIFAR-6-150), seed = 1  
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_cifar10_150_1.yaml

# CIFAR100, seen/unseen split of 50/50, 4 labels per seen class (CIFAR-50-200), seed = 1  
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_cifar100_200_1.yaml

# CIFAR100, seen/unseen split of 80/20, 4 labels per seen class (CIFAR-80-320), seed = 1    
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_cifar100_320_1.yaml

# ImageNet30, seen/unseen split of 20/10, 1% labeled data (ImageNet-20-p1), seed = 1  
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_in30_p1_1.yaml
```

### Evaluation

After training, the best checkpoints will be saved in ``./saved_models``. The closed-set performance has been reported in the training logs. For the open-set evaluation, please see [``evaluate.ipynb``](./evaluate.ipynb).


## Acknowledgments

We sincerely thank the authors of [IOMatch (ICCV'23)](https://github.com/nukezil/IOMatch) for creating such an awesome SSL benchmark.


## Citation

```bibtex
@INPROCEEDINGS{10981231,
  author={Kareem, Daniya Najiha A. and Lahoud, Jean and Fiaz, Mustansar and Kumar, Amandeep and Cholakkal, Hisham},
  booktitle={2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)}, 
  title={Open-Set Semi-Supervised Learning for Long-Tailed Medical Datasets}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Ethics;Heavily-tailed distribution;Image recognition;Open Access;Conferences;Training data;Skin;Data models;Standards;Biomedical imaging},
  doi={10.1109/ISBI60581.2025.10981231}}
```
