这个版本改成 Linux 版本，需要更改的内容：

- 导入包
!pip install spectral
from google.colab import drive
drive.mount('/content/drive')

- 添加环境变量
sys.path.append('/content/drive/MyDrive/code/MDL/')




Code and Paper source: 
<!-- - [EndNet](https://github.com/danfenghong/IEEE_GRSL_EndNet)
- [MDL](https://github.com/danfenghong/IEEE_TGRS_MDL-RS)
- [HCTNet](https://github.com/zgr6010/Fusion_HCT) --------------------------------- [Paper](https://ieeexplore.ieee.org/document/9999457)
- [FusAtNet](https://github.com/ShivamP1993/FusAtNet-Dual-Attention-based-SpectroSpatial-Multimodal-Fusion-Network-for-Hyperspectral-and-LiDAR-)
- [S2ENet](https://github.com/likyoo/Multimodal-Remote-Sensing-Toolkit)
- [Cross-HL](https://github.com/AtriSukul1508/Cross-HL)-------------------------------- [Paper](https://ieeexplore-ieee-org.elibrary.jcu.edu.au/document/10462184)
- [MIViT](https://github.com/icey-zhang/MIViT)
- [MS2CANet](https://github.com/junhengzhu/MS2CANet)
- [SHNet](https://github.com/quanweiliu/SHNet)
--> 


Code and Paper source: 
| Code      | Paper |  Journal |  Year | 
| ----------- | ----------- |----------- |----------- |
| [EndNet](https://github.com/danfenghong/IEEE_GRSL_EndNet)      | Title       | Title       | Title       |
| [MDL](https://github.com/danfenghong/IEEE_TGRS_MDL-RS)   | Text        | Title       |
| [HCTNet](https://github.com/zgr6010/Fusion_HCT)   | [Paper](https://ieeexplore.ieee.org/document/9999457)        | TGRS       | 2022       |
| [FusAtNet](https://github.com/ShivamP1993/FusAtNet-Dual-Attention-based-SpectroSpatial-Multimodal-Fusion-Network-for-Hyperspectral-and-LiDAR-)   | Text        | CVPR        | 2020       |
| [S2ENet](https://github.com/likyoo/Multimodal-Remote-Sensing-Toolkit)   | Text        | Title       | Title       |
| [Cross-HL](https://github.com/AtriSukul1508/Cross-HL)  | [Paper](https://ieeexplore-ieee-org.elibrary.jcu.edu.au/document/10462184)        | Title       | Title       |
| [MIViT](https://github.com/icey-zhang/MIViT)   | Text        | Title       | Title       |
| [MS2CANet](https://github.com/junhengzhu/MS2CANet)   | Text        | Title       | Title       |
| [SHNet](https://github.com/quanweiliu/SHNet)   | Text        | Title       | Title       |


This repository proposed a new taxonomy to descibe existed patch based image classification (semantic segmenation) models.

Based the input and output of the model, we categorized these pixel-level classification model into singlesacle singlemodality input and singleoutput (SSISO), singlesacle multimodelity input and singleoutput (SMISO), singlesacle multimodelity input and multioutput (SMIMO), mutlisacle multimodelity input and singleoutput (MMISO), mutlisacle multimodelity input and multiouput (MMIMO).

Of course, there are multiscale singlemodality input, singleoutput (MSISO) and multiscale singlemodality input and multioutput (MSIMO) and so on. We will continue and add them in this framework.

I have collected a range of models based this taxonomy. If you want to contribute this repository and make it better, feel free to contact me. My emial : quanwei.liu@my.jcu.edu.au



Noting: 
- MS2CANet : 原始的代码
- MS2CANet2 : 将代码拆分了两部分，测试精度，结果是一模一样的
