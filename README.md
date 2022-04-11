# [CVPR 2021] Contrastive Neural Architecture Search with Neural Architecture Comparators [[PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Contrastive_Neural_Architecture_Search_With_Neural_Architecture_Comparators_CVPR_2021_paper.pdf)]

## Introduction

One of the key steps in Neural Architecture Search (NAS) is to estimate the performance of candidate architectures. Existing methods either directly use the validation performance or learn a predictor to estimate the performance. However, these methods can be either computationally expensive or very inaccurate, which may severely affect the search efficiency and performance. Moreover, as it is very difficult to annotate architectures with accurate performance on specific tasks, learning a promising performance predictor is often non-trivial due to the lack of labeled data. In this paper, we argue that it may not be necessary to estimate the absolute performance for NAS. On the contrary, we may need only to understand whether an architecture is better than a baseline one. However, how to exploit this comparison information as the reward and how to well use the limited labeled data remains two great challenges. In this paper, we propose a novel Contrastive Neural Architecture Search (CTNAS) method which performs architecture search by taking the comparison results between architectures as the reward. Specifically, we design and learn a Neural Architecture Comparator (NAC) to compute the probability of candidate architectures being better than a baseline one. Moreover, we present a baseline updating scheme to improve the baseline iteratively in a curriculum learning manner. More critically, we theoretically show that learning NAC is equivalent to optimizing the ranking over architectures. Extensive experiments in three search spaces demonstrate the superiority of our CTNAS over existing methods.

<p align="center">
<img src="overview.png" alt="Contrastive Neural Architecture Search" width="90%" align=center />
</p>

## Requirements

Please install all the requirements in `requirements.txt`.

## Training Method


First, we need to **download the architecture-accuracy pairs data**.
```
wget https://github.com/chenyaofo/CTNAS/releases/download/data/nas_bench.json -O ctnas/data/nas_bench.json
```

**Train on NAS-Bench-101**

```
python ctnas/train.py --space nasbench --data ctnas/data/nas_bench.json --train_batch_size 256 --output output/nasbench_search
```

## Citation

If you use any part of our code in your research, please cite our paper:

```BibTex
@InProceedings{chen2021contrastive,
  title = {Contrastive Neural Architecture Search with Neural Architecture Comparators},
  author = {Yaofo Chen and Yong Guo and Qi Chen and Minli Li and Yaowei Wang and Wei Zeng and Mingkui Tan},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2021}
}
```
