## Learning to Filter: Siamese Relation Network for Robust Tracking  

* **Abstract**: Despite the great success of Siamese-based trackers, their performance under complicated scenarios is still not satisfying, especially when there are distractors. To this end, we propose a novel Siamese relation network, which introduces two efficient modules, i.e. Relation Detector (RD) and Refinement Module (RM). RD performs in a meta-learning way to obtain a learning ability to filter the distractors from the background while RM aims to effectively integrate the proposed RD into the Siamese framework to generate accurate tracking result. Moreover, to further improve the discriminability and robustness of the tracker, we introduce a contrastive training strategy that attempts not only to learn matching the same target but also to learn how to distinguish the different objects. Therefore, our tracker can achieve accurate tracking results when facing background clutters, fast motion, and occlusion. Experimental results on five popular benchmarks, including VOT2018, VOT2019, OTB100, LaSOT, and UAV123, show that the proposed method is effective and can achieve state-of-the-art results.

## Citation
```
@InProceedings{Cheng_2021_CVPR,
    author    = {Cheng, Siyuan and Zhong, Bineng and Li, Guorong and Liu, Xin and Tang, Zhenjun and Li, Xianxian and Wang, Jing},
    title     = {Learning To Filter: Siamese Relation Network for Robust Tracking},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {4421-4431}
}
```
## Paper and Result
The full paper is available [here](https://arxiv.org/abs/2104.00829).  

The raw results are [here](https://drive.google.com/drive/folders/1NfLcvUUcTIdMSDWGa1PXKL9D-dM4bi4J?usp=sharing) or [here](https://pan.baidu.com/s/1lG7uq5GHGEpfRj-qNpOu4Q) (extraction code: `bw3b`) for comparison.  

The code based on the  [SiamBAN](https://github.com/hqucv/siamban)   


<div align="center">
  <img src="demo/box.gif" width="640px" />
  <p>Examples of SiamRN outputs. The green boxes are the results yielded by SiamRN, and other boxes are the results yileded by DiMP and SiamBAN respectively.</p>
</div>

## Using
The usage of this repo is similar with PYSOT and SiamBAN, including data pre-processing, training, testing and tuning, refer to [here](https://github.com/hqucv/siamban).

## Contact
If you have any question, please feel free to contact with me.  
E-mail: siyuancheng1019@gmail.com
