# Learning Deep Multimodal Feature Representation with Asymmetric Multi-layer Fusion

By Yikai Wang, Fuchun Sun, Ming Lu, Anbang Yao.


## Datasets

For semantic segmentation task on NYUDv2 ([official dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)), we provide a link to download the dataset [here](https://drive.google.com/drive/folders/1mXmOXVsd5l9-gYHk92Wpn6AcKAbE0m3X?usp=sharing). The provided dataset is originally preprocessed in this [repository](https://github.com/DrSleep/light-weight-refinenet), and we add depth data in it. Please modify the data paths in the codes, where we add comments 'Modify data path'.


## Dependencies
```
python==3.6.2
pytorch==1.0.0
torchvision==0.2.2
imageio==2.4.1
numpy==1.16.2
scikit-learn==0.20.2
scipy==1.1.0
opencv-python==4.0.0
```


## Scripts

First, 
```
cd semantic_segmentation
```
Training script for segmentation with RGB and Depth input, the default setting uses RefineNet (ResNet101),
```
python main.py --gpu 0 -c exp_name  # or --gpu 0 1 2
```
Evaluation script,
```
python main.py --gpu 0 --resume path_to_pth --evaluate  # optionally use --save-img to visualize results
```

## License

AsymFusion is released under MIT License.


## Citation
If you find our work useful for your research, please consider citing the following paper.
```
@inproceedings{wang2020asymfusion,
  title={Learning Deep Multimodal Feature Representation with Asymmetric Multi-layer Fusion},
  author={Wang, Yikai and Sun, Fuchun and Lu, Ming and Yao, Anbang},
  booktitle={ACM International Conference on Multimedia (ACM MM)},
  year={2020}
}
```