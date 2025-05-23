# HSI Classification and HSI, SAR, LiDAR Classification

# Summary
| Code      | Paper |  Journal |  Year | 
| ----------- | ----------- |----------- |----------- |
||**Single modality models**||
| [FDGC](https://github.com/quanweiliu/FDGC)      | [A Fast Dynamic Graph Convolutional Network and CNN Parallel Network for Hyperspectral Image Classification](https://ieeexplore-ieee-org.elibrary.jcu.edu.au/abstract/document/9785802)       | TGRS       | 2022       | 
| [SSFTT](https://github.com/zgr6010/HSI_SSFTT) |[Spectral–Spatial Feature Tokenization Transformer for Hyperspectral Image Classification](https://ieeexplore-ieee-org.elibrary.jcu.edu.au/document/9684381/)| TGRS | 2022  |
| [morphFormer](https://github.com/mhaut/morphFormer)      | [Spectral–Spatial Morphological Attention Transformer for Hyperspectral Image Classification](https://ieeexplore-ieee-org.elibrary.jcu.edu.au/document/10036472)       | TGRS       | 2023      
| [ViT-DGCN](https://github.com/quanweiliu/ViT-DGCN)      | [Fusion of GaoFen-5 and Sentinel-2B data for lithological mapping using vision transformer dynamic graph convolutional network](https://www.sciencedirect.com/science/article/pii/S1569843224001341)       | JAG       | 2024       |
| [DBCTNet](https://github.com/xurui-joei/DBCTNet)      | [DBCTNet: Double Branch Convolution-Transformer Network for Hyperspectral Image Classification](https://ieeexplore-ieee-org.elibrary.jcu.edu.au/document/10440601)       | TGRS       | 2024       | 
||**Multimodal models**||
| [EndNet](https://github.com/danfenghong/IEEE_GRSL_EndNet)      | [Deep Encoder-Decoder Networks for Classification of Hyperspectral and LiDAR Data](https://ieeexplore-ieee-org.elibrary.jcu.edu.au/document/9179756)       | TGRSL       | 2020       |
| [MDL](https://github.com/danfenghong/IEEE_TGRS_MDL-RS)   | [More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classification](https://ieeexplore.ieee.org/document/9174822)        | TGRS       |  2021 |
| [HCTNet](https://github.com/zgr6010/Fusion_HCT)   | [Spectral–Spatial Feature Tokenization Transformer for Hyperspectral Image Classification](https://ieeexplore.ieee.org/document/9999457)        | TGRS       | 2022       |
| [FusAtNet](https://github.com/ShivamP1993/FusAtNet-Dual-Attention-based-SpectroSpatial-Multimodal-Fusion-Network-for-Hyperspectral-and-LiDAR-)   | [FusAtNet: Dual Attention based SpectroSpatial Multimodal Fusion Network for Hyperspectral and LiDAR Classification](https://ieeexplore-ieee-org.elibrary.jcu.edu.au/document/9150738)        | CVPR        | 2020       |
| [S2ENet](https://github.com/likyoo/Multimodal-Remote-Sensing-Toolkit)   | [S²ENet: Spatial–Spectral Cross-Modal Enhancement Network for Classification of Hyperspectral and LiDAR Data]()        | TGRSL       | 2022       |
| [Cross-HL](https://github.com/AtriSukul1508/Cross-HL)  | [Cross Hyperspectral and LiDAR Attention Transformer: An Extended Self-Attention for Land Use and Land Cover Classification](https://ieeexplore-ieee-org.elibrary.jcu.edu.au/document/10462184)        | TGRS       | 2024       |
| [MIViT](https://github.com/icey-zhang/MIViT)   | [Multimodal Informative ViT: Information Aggregation and Distribution for Hyperspectral and LiDAR Classification](https://ieeexplore-ieee-org.elibrary.jcu.edu.au/document/10464367)        | TCSVT       | 2024       |
| [MS2CANet](https://github.com/junhengzhu/MS2CANet)   | [MS2CANet: Multiscale Spatial–Spectral Cross-Modal Attention Network for Hyperspectral Image and LiDAR Classification](https://ieeexplore-ieee-org.elibrary.jcu.edu.au/document/10382694/)        | TGRSL       | 2024       | 
| [SHNet](https://github.com/quanweiliu/SHNet)   | Enhancing Oil Spill Detection with Controlled Random Sampling: A Multimodal Fusion Approach Using SAR and HSI Imagery        | RSA       | 2025       |

# Description
This repository proposed a new taxonomy to descibe existed patch based image classification (semantic segmenation) models.

Based the input and output of the model, we categorized these pixel-level classification model into singlesacle singlemodality input and singleoutput (SSISO), singlesacle multimodelity input and singleoutput (SMISO), singlesacle multimodelity input and multioutput (SMIMO), mutlisacle multimodelity input and singleoutput (MMISO), mutlisacle multimodelity input and multiouput (MMIMO).

Of course, there are multiscale singlemodality input, singleoutput (MSISO) and multiscale singlemodality input and multioutput (MSIMO) and so on. We will continue and add them in this framework.


# Nomalization:
1. MinMax normalization
$$
x_i' = \frac{x_i - \min(x)}{\max(x) - \min(x)}
$$

**特点**：
- 操作单位：每个通道（维度）
- 输出范围固定（如 [0, 1]）
- 保留比例信息（但不保留方向信息）
- 对极值敏感

2. Unit-norm normalization
$$
\mathbf{x'} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2} = \frac{\mathbf{x}}{\sqrt{\sum_i x_i^2}}
$$

**特点**：
- 操作单位：每个样本/像素向量
- 所有向量模长为 1，分量值通常在 [-1, 1] 范围
- 强调方向信息（如光谱形状）
- 不保留幅值大小
> 在高光谱图像处理任务中，单位范数归一化更适合注重“光谱形状”的任务；而 Min-Max 更适合需要统一数值尺度或可视化展示的场景

# Tricks
- 添加环境变量

```pyhon
import sys
sys.path.append('/content/drive/MyDrive/code/MDL/')
```
- 更改所用的 GPU
```pyhon
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [0]))
```


Noting: 
- MS2CANet : 原始的代码
- MS2CANet2 : 将代码拆分了两部分，测试精度，结果是一模一样的



I have collected a range of models based this taxonomy. If you want to contribute this repository and make it better, feel free to contact me. My emial : quanwei.liu@my.jcu.edu.au

