# QKFormer: Hierarchical Spiking Transformer using Q-K Attention [NeurIPS 2024](https://arxiv.org/abs/2403.16552)


QKFormer achieves **a groundbreaking top-1 accuracy of **85.65%** on ImageNet-1k**, which is the first time that directly training SNNs have exceeded 85% accuracy on ImageNet-1K in 2024/03.


<p align="center">
<img src="https://github.com/zhouchenlin2096/QKFormer/blob/master/imgs/QKFormer.png">
</p>

## Main results on ImageNet-1K

| Model                | Resolution| T      |  Type    | Param.      | Top-1 Acc (%)| Download |
| :---:                | :---:     | :---:  |:---:  | :---:       |:---:      |:---:      |
| ViT                  | 384x384   | -      | ANN   |  85.9M     |  77.9    |   -       |
| Deit-B               | 384x384   | -      | ANN   |  86.0M     |  83.1    |   -       |
| Swin transformer     | 384x384   | -      | ANN   |  88.0M     |  84.5    |   -       |
| Spikformer-8-768     | 224x224   | 4      | SNN   |  66.34M     |  74.81    |   -       |
| Spikingformer-8-768  | 224x224   | 4      | SNN   |  66.34M     |  75.85    |   -       |
| QKFormer-10-384     | 224x224   | 4      | SNN   | 16.47M     |  78.80    |   -       |
| QKFormer-10-512     | 224x224   | 4      | SNN   | 29.08M     |  82.04     |     -     |
| QKFormer-10-768     | 224x224   | 4      | SNN   |  64.96M     |   84.22    |   -        | 
| QKFormer-10-768     | 288x288   | 4     | SNN   |  64.96M     |   85.25   |   -        | 
| QKFormer-10-768     | 384x384   | 4      | SNN   |  64.96M     |  **85.65**   |   -        | 

## News

[2024.10.10] Update code and trained models.

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
| Spikingformer-8-768 | 224x224   | 4  |  66.34M     | 12.54G  | 13.68 mJ  |75.85  |   [here](https://pan.baidu.com/s/1LsECpFOxh30O3vHWow8OGQ) |

All download passwords: abcd


## Main results on CIFAR10/CIFAR100/CIFAR10-DVS/DVS128

| Model                | T      |  Param.     | CIFAR10 Top-1 Acc| Download  |CIFAR100 Top-1 Acc|
| :---:                | :---:  | :---:       |  :---:  |:---:   |:---: |
| Spikingformer-4-256  | 4      |  4.15M     | 94.77   |   -   |77.43  |
| Spikingformer-2-384  | 4      |  5.76M     | 95.22   |   -   |78.34  |
| Spikingformer-4-384  | 4      |  9.32M     | 95.61    |   -  |79.09  |
| Spikingformer-4-384-400E  | 4      |  9.32M     | 95.81    | - |79.21  |


## Requirements
timm==0.6.12; cupy==11.4.0; torch==1.12.1; spikingjelly==0.0.0.0.12; pyyaml; tensorboard; 

data prepare: ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Train
### Training  on ImageNet
Setting hyper-parameters in imagenet.yml

```
cd imagenet
python -m torch.distributed.launch --nproc_per_node=8 train.py
```

### Testing ImageNet Val data
Download the trained model first [here](https://pan.baidu.com/s/1LsECpFOxh30O3vHWow8OGQ), passwords: abcd
```
cd imagenet
python test.py
```

### Training  on CIFAR10
Setting hyper-parameters in cifar10.yml
```
cd cifar10
python train.py
```

### Training  on CIFAR100
Setting hyper-parameters in cifar100.yml
```
cd cifar10
python train.py
```

### Training  on DVS128 Gesture
```
cd dvs128-gesture
python train.py
```

### Training  on CIFAR10-DVS
```
cd cifar10-dvs
python train.py
```

## Acknowledgement & Contact Information
Related project: [spikformer](https://github.com/ZK-Zhou/spikformer), [pytorch-image-models](https://github.com/huggingface/pytorch-image-models), [spikingjelly](https://github.com/fangwei123456/spikingjelly).

For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please get in touch with zhouchl@pcl.ac.cn or zhouchenlin19@mails.ucas.ac.cn.
