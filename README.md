## AnimeCeleb &mdash; Official Dataset & PyTorch Implementation

![Teaser image](./assets/teaser.png)

> **AnimeCeleb: Large-Scale Animation CelebHeads Dataset for Head Reenactment**<br>
> [Kangyeol Kim](https://kangyeolk.github.io/)\*<sup>1,4</sup>, [Sunghyun Park](https://psh01087.github.io)\*<sup>1</sup>, [Jaeseong Lee](https://leejesse.github.io/)\*<sup>1</sup>, [Sunghyo Chung](https://sunghyo.github.io/)<sup>2</sup>, [Junsoo Lee](https://ssuhan.github.io/)<sup>3</sup>, [Jaegul Choo](https://sites.google.com/site/jaegulchoo)<sup>1</sup><br>
> <sup>1</sup>KAIST, <sup>2</sup>Korea University, <sup>3</sup>Naver Webtoon, <sup>4</sup>Letsur Inc.<br>
> In ECCV 2022. (* indicates equal contribution)

> Paper: https://arxiv.org/abs/2111.07640 <br>
<!-- > Project page: TBD <br> -->

> **Abstract:** *We present a novel Animation CelebHeads dataset (AnimeCeleb) to address an animation head reenactment. Different from previous animation head datasets, we utilize a 3D animation models as the controllable image samplers, which can provide a large amount of head images with their corresponding detailed pose annotations. To facilitate a data creation process, we build a semi-automatic pipeline leveraging an open 3D computer graphics software with a developed annotation system. After training with the AnimeCeleb, recent head reenactment models produce high-quality animation head reenactment results, which are not achievable with existing datasets. Furthermore, motivated by metaverse application, we propose a novel pose mapping method and architecture to tackle a cross-domain head reenactment task. During inference, a user can easily transfer one's motion to an arbitrary animation head. Experiments demonstrate an usefulness of the AnimeCeleb to train animation head reenactment models, and the superiority of our cross-domain head reenactment model compared to state-of-the-art methods.*

## TL;DR
This repository consists of 3 parts as follows:
* Downloadable links of AnimeCeleb ([click here](https://forms.gle/wN1d6kNZv6sn6ad66)) and author list ([click here](https://drive.google.com/file/d/1N9hIshJ_gQcFmVeelmQhl7-j_zlyTcwN/view?usp=sharing)).
* Annotation tool to create other images with a 3D model and Blender software (Release Soon).
* A source code of the proposed algorithm for cross-domain head reenactment ([click here](./Animo/)). 


## Citation
If you find this work useful for your research, please cite our paper:

```
@inproceedings{kim2021animeceleb,
  title={AnimeCeleb: Large-Scale Animation CelebHeads Dataset for Head Reenactment},
  author={Kim, Kangyeol and Park, Sunghyun and Lee, Jaeseong and Chung, Sunghyo and Lee, Junsoo and Choo, Jaegul},
  booktitle={Proc. of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```


## Acknowledgments

We appreciate other outstanding projects: series of [talking-head-anime](https://github.com/pkhungurn), [Making Anime faces with StyleGan](https://www.gwern.net/Faces#stylegan-2) that inspired us.
Also, we would like to thank the original authors of the collected 3D model, and open the list of their names and URLs in this [file](https://drive.google.com/file/d/1N9hIshJ_gQcFmVeelmQhl7-j_zlyTcwN/view?usp=sharing).
The model code borrows heavily from [FOMM](https://github.com/AliaksandrSiarohin/first-order-model) and [PIRenderer](https://github.com/RenYurui/PIRender).
