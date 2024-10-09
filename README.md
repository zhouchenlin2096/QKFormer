# QKFormer: Hierarchical Spiking Transformer using Q-K Attention ([NeurIPS 2024](https://arxiv.org/abs/2403.16552))

QKFormer achieves **a groundbreaking top-1 accuracy of **85.65%** on ImageNet-1k**, the first time **directly training SNNs** have exceeded 85% accuracy on ImageNet-1K in 2024/03.


<p align="center">
<img src="https://github.com/zhouchenlin2096/QKFormer/blob/master/imgs/QKFormer.png">
</p>


## News

[2024.10.10] Update code and trained models.

[2024.09.25] Accepted as a spotlight in NeurIPS 2024.


## Main results on ImageNet-1K

| Model                |  Type|  Architecture  | Resolution| T        | Param.      | Top-1 Acc (%)| Download |
| :---:                |:---: |:---: | :---:     | :---:    | :---:       |:---:      |:---:      |
| ViT                  | ANN | ViT-B/16| 384x384   | -         |  85.9M     |  77.9    |   -       |
| Deit                 | ANN | DeiT-B | 384x384   | -         |  86.0M     |  83.1    |   -       |
| Swin transformer     | ANN | Swin Transformer-B | 384x384   | -        |  88.0M     |  84.5    |   -       |
| SEW-ResNet           | SNN | SEW-ResNet-152 | 224x224   | 4         |  60.19M     |  69.26    |   -       |
| Spikformer           | SNN | Spikformer-8-768 | 224x224   | 4         |  66.34M     |  74.81    |   -       |
| Spikingformer        | SNN | Spikingformer-8-768 | 224x224   | 4        |  66.34M     |  75.85    |   -       |
| **QKFormer**             | SNN | HST-10-384 | 224x224   | 4         | 16.47M     |  **78.80**      |   [link](https://pan.baidu.com/s/1mX0jQyKZ5p6ZDzvMVeY20A)   |
| **QKFormer**             | SNN | HST-10-512 | 224x224   | 4         | 29.08M     |  **82.04**      |     [link](https://pan.baidu.com/s/1luWM1L8gV3BI7REh4MgbkA)    |
| **QKFormer**             | SNN | HST-10-768 | 224x224   | 4         |  64.96M     |   **84.22**    |   [link](https://pan.baidu.com/s/1WJW1wC0Vs-lvGjYr5pGV_w)        | 
| **QKFormer**             | SNN | HST-10-768 | 288x288   | 4         |  64.96M     |   **85.25**     |   [link](https://pan.baidu.com/s/1UaqY98UqJPJbosKfY103Jg)      | 
| **QKFormer**             | SNN | HST-10-768 | 384x384   | 4         |  64.96M     |  **85.65**  |   [link](https://pan.baidu.com/s/1gRAZR9gkMr5ScHK-kwZAnw)        | 

All download passwords: **abcd**


## Requirements

```
timm==0.6.12
cupy==11.4.0
torch==1.12.1
spikingjelly==0.0.0.0.12
pyyaml
tensorboard
```

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

## Train & Test
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

## Reference
If you find this repo useful, please consider citing:
```
@article{zhou2024qkformer,
  title={QKFormer: Hierarchical Spiking Transformer using QK Attention},
  author={Zhou, Chenlin and Zhang, Han and Zhou, Zhaokun and Yu, Liutao and Huang, Liwei and Fan, Xiaopeng and Yuan, Li and Ma, Zhengyu and Zhou, Huihui and Tian, Yonghong},
  journal={arXiv preprint arXiv:2403.16552},
  year={2024}
}
```


## Acknowledgement & Contact Information
Related project: [spikformer](https://github.com/ZK-Zhou/spikformer), [spikingformer](https://github.com/zhouchenlin2096/Spikingformer), [spikingjelly](https://github.com/fangwei123456/spikingjelly).

For help or issues using this git, please submit a GitHub issue. 

For other communications related to this git, please contact zhouchl@pcl.ac.cn or zhouchenlin19@mails.ucas.ac.cn.
