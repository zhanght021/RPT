# RPT: Learning Point Set Representation for Siamese Visual Tracking[[ECCVW2020](https://arxiv.org/abs/2008.03467)]


## :sunny: Currently, this code only supports the offline version of RPT, the absolute RPT code will come soon!


## News
- :trophy: **We are the Winner of VOT-2020 Short-Term challenge**
- :trophy: **We get both 1st on the public and sequestered benchmark dataset of the VOT2020 Short-Term challenge**
- :sunny::sunny:**Our [VOT2020-ST Winner presentation](https://github.com/zhanght021/RPT/blob/master/VOT-ST2020%2BWinners%2BPresentation.pdf) has been uploaded**


----
## Spotlight video

[![Video Label](https://i0.hdslb.com/bfs/album/1ea9e961083d81f7fed53d22ed8698a1ac2307f9.jpg@518w_1e_1c.jpg)](https://www.bilibili.com/video/BV17v41117cZ)


---
## Models
| Dataset | pattern | A | R | EAO | Config. Filename |
|:---:|:---:|:---:|:---:|:---:|:---:|
| VOT2018 | offline | 0.610 | 0.150 | 0.497 | config_vot2018_offline.yaml |
| VOT2019 | offline | 0.598 | 0.261 | 0.409 | config_vot2019_offline.yaml |
| VOT2018 | online | 0.629 | 0.103 | 0.510 | :smile:coming soon:smile: |
| VOT2019 | online | 0.623 | 0.186 | 0.417 | :smile:coming soon:smile: |

- The pretrained model can be downloaded [[here](https://pan.baidu.com/s/18EXDr4DoeD89Vasuf8WCXQ)], extraction code: g4ac.
- The raw results can be downloaded [[here](https://pan.baidu.com/s/1fAovMOR8UAN46f5Dm-sa6A)], extraction code: mkbh.

----
## Abstract
While remarkable progress has been made in robust visual tracking, accurate target state estimation still remains a highly challenging problem. In this paper, we argue that this issue is closely related to the prevalent bounding box representation, which provides only a coarse spatial extent of object. Thus an effcient visual tracking framework is proposed to accurately estimate the target state with a finer representation as a set of representative points. The point set is trained to indicate the semantically and geometrically significant positions of target region, enabling more fine-grained localization and modeling of object appearance. We further propose a multi-level aggregation strategy to obtain detailed structure information by fusing hierarchical convolution layers. Extensive experiments on several challenging benchmarks including OTB2015, VOT2018, VOT2019 and GOT-10k demonstrate that our method achieves new state-of-the-art performance while running at over 20 FPS.

---
## Installation
Please find installation instructions in INSTALL.md

---
## Quick Start: Using siamreppoints

Download pretrained models and put the siamreppoints.model in the correct directory in experiments

```bash
cd siamreppoints/tools
python test.py \
       --snapshot ./snapshot/siamreppoints.model \  #model path
       --dataset VOT2018 \  #dataset name
       --config ./experiments/siamreppoints/config_vot2018_offline.yaml  #config file 
```


```bash
cd siamreppoints/tools
python eval.py \
       --tracker_path ./results \  #result path
       --dataset VOT2018 \  #dataset name
       --tracker_prefix 'siam' \  # tracker_name
       --num 1  # number thread to eval
```

---
## Ackowledgement
- [pysot](https://github.com/STVIR/pysot)
