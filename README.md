# Hierarchical Spiking Transformer using Q-K Attention [This link](https://arxiv.org/pdf/2403.16552.pdf)



QKFormer achieves a groundbreaking top-1 accuracy of **85.65% top-1** on ImageNet-1k, which is the first time that directly training SNNs have exceeded 85% accuracy on ImageNet-1K in 2024/03.

## Reference
If you find this repo useful, please consider citing:
```
@article{zhou2023spikingformer,
  title={Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network},
  author={Zhou, Chenlin and Yu, Liutao and Zhou, Zhaokun and Zhang, Han and Ma, Zhengyu and Zhou, Huihui and Tian, Yonghong},
  journal={arXiv preprint arXiv:2304.11954},
  year={2023},
  url={https://arxiv.org/abs/2304.11954}
}
```

## Main results on ImageNet-1K

| Model               | Resolution| T |  Param.     | FLOPs   |  Power |Top-1 Acc| Download |
| :---:               | :---:     | :---:  | :---:       |  :---:  |  :---:    |:---: |:---: |
| Spikingformer-8-384 | 224x224   | 4 |  16.81M     | 3.88G   | 4.69 mJ   |72.45  |   -    |
| Spikingformer-8-512 | 224x224   | 4 |  29.68M     | 6.52G  | 7.46 mJ   |74.79  |     -  |
| Spikingformer-8-768 | 224x224   | 4  |  66.34M     | 12.54G  | 13.68 mJ  |75.85  |   - |
