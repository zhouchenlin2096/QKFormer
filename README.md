# Hierarchical Spiking Transformer using Q-K Attention [This link](https://arxiv.org/pdf/2403.16552.pdf)


QKFormer achieves a groundbreaking top-1 accuracy of **85.65% top-1** on ImageNet-1k, which is the first time that directly training SNNs have exceeded 85% accuracy on ImageNet-1K in 2024/03.


## Main results on ImageNet-1K

| Model                | Resolution| T      |  Type    | Param.      | Top-1 Acc| Download |
| :---:                | :---:     | :---:  |:---:  | :---:       |:---:      |:---:      |
| ViT                  | 384x384   | -      | ANN   |  85.9M     |  77.9    |   -       |
| Deit-B               | 384x384   | -      | ANN   |  86.0M     |  83.1    |   -       |
| Swin transformer     | 384x384   | -      | ANN   |  88.0M     |  84.5    |   -       |
| Spikingformer-8-768  | 224x224   | 4      | SNN   |  66.34M     |  75.85    |   -       |
| Spikingformer-8-768  | 224x224   | 4      | SNN   |  66.34M     |  75.85    |   -       |
| QKFormer-10-384     | 224x224   | 4      | SNN   | 16.47M     |  78.80    |   -       |
| QKFormer-10-512     | 224x224   | 4      | SNN   | 29.08M     |  82.04     |     -     |
| QKFormer-10-768     | 224x224   | 4      | SNN   |  64.96M     |   84.22    |   -        | 
| QKFormer-10-768     | 288x288   | 4     | SNN   |  64.96M     |   85.25   |   -        | 
| QKFormer-10-768     | 384x384   | 4      | SNN   |  64.96M     |  **85.65**   |   -        | 
