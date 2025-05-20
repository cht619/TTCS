### Test-Time Medical Image Segmentation Using CLIP-Guided SAM Adaptation


[![IEEE Paper](https://img.shields.io/badge/-IEEE%20Paper-0072C6?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/10822570)


### Requirements

  ```bash
  mmcv > 2.0
  torch
  ```

### Datasets
Downloads: [Funds](https://github.com/Chen-Ziyang/VPTTA?tab=readme-ov-file)


#### Run
The core codes of [TTCS](./mmseg/baselines/clip_tta.py#L249-L253).
  ```bash
  bash TTA/exp/tta/train_fundus.sh 0 RIM_ONE_r3 TTCS
  ```



## Citation

If this codebase is useful for your work, please cite the following papers:

```BibTeX
@inproceedings{chen2024test,
  title={Test-Time Medical Image Segmentation Using CLIP-Guided SAM Adaptation},
  author={Chen, Haotian and Xu, Yonghui and Xu, Yanyu and Zhang, Yixin and Cui, Lizhen},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={1866--1873},
  year={2024},
  organization={IEEE}
}
```
