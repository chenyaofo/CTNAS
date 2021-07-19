# Contrastive Neural Architecture Search with Neural Architecture Comparators [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Contrastive_Neural_Architecture_Search_With_Neural_Architecture_Comparators_CVPR_2021_paper.pdf)]

Pytorch Implementation for "Contrastive Neural Architecture Search with Neural Architecture Comparators".

<p align="center">
<img src="overview.png" alt="Contrastive Neural Architecture Search" width="90%" align=center />
</p>

**This is a pre release code, which would be optmized in the next few days.**

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
