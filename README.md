
# [AAAI 2023] Semi-supervised Deep Large-baseline Homography Estimation with Progressive Equivalence Constraint. [[Paper]](https://arxiv.org/abs/2212.02763).
<h4 align="center">Hai Jiang<sup>1,2</sup>, Haipeng Li<sup>2,3</sup>, Yuhang Lu<sup>4</sup>, Songchen Han<sup>1</sup>, Shuaicheng Liu<sup>2,3</sup></center>
<h4 align="center">1.Sichuan University, 2.University of Electronic Science and Technology of China, 
<h4 align="center">3.Megvii Technology, 4.University of South Carolina</center></center>

## Presentation video:  
[[Youtube]](https://www.youtube.com/watch?v=-ktR2mwq5H4) and [[Bilibili]](https://www.bilibili.com/video/BV13Y41117K3/?vd_source=225fbd2f43ab4ac27f3ec5e9f87dd029)
## Pipeline
![](https://github.com/megvii-research/LBHomo/blob/main/Figs/Pipeline.jpg)
## Dependencies
```
pip install -r requirements.txt
````
- This repo includes GOCor as git submodule. You need to pull submodules with
```
git submodule update --init --recursive
git submodule update --recursive --remote
```

## Download the raw dataset
Please refer to [Content-Aware Unsupervised Deep Homography Estimation.](https://github.com/JirongZhang/DeepHomography).

- Dataset download links: [[GoogleDriver]](https://drive.google.com/file/d/19d2ylBUPcMQBb_MNBBGl9rCAS7SU-oGm/view?usp=sharing), [[BaiduYun]](https://pan.baidu.com/s/1Dkmz4MEzMtBx-T7nG0ORqA) (key:gvor)

- Unzip the data to the directory "./dataset"

- Run "video2img.py"
```
Be sure to scale the image to (640, 360) since the point coordinate system is based on the (640, 360).
e.g. img = cv2.imresize(img, (640, 360))
```
- Using the images in "train.txt" and "test.txt" for training and evaluation, the manually labeled evaluation files can be downloaded from: [[GoogleDriver]](https://drive.google.com/drive/folders/1Fwe0TnaKB7FudJu_PLcsq8765WEs7DAG?usp=sharing), [[BaiduYun]](https://pan.baidu.com/s/1xd6Q9P94lSE7021yanIALQ)(key:i721).
## Pre-trained model
```
The model provided below is the retrained version(with some differences in reported results)
```

| model    | RE | LT | LL | LF | SF | Avg | Model |
| --------- | ----------- | ------------ |------------ |------------ |------------ |------------ |------------ |
| Pre-trained | 1.66 | 5.49 | 4.11 | 7.57 | 6.95  | 5.16  |[[Baidu]](https://pan.baidu.com/s/1QZ-NJ4WMK7dEZb63c3Q2vQ)(code: 0jfu) [[Google]](https://drive.google.com/file/d/1SQWOYdjsDt9U3jMSNyLF2oDqB4w9aX1p/view?usp=sharing)
## How to train?
You need to modify ```dataset/data_loader.py``` slightly for your environment, and then
```
python train.py --model_dir experiments/base_model/ 
```
## How to test?
```
python evaluate.py --model_dir experiments/base_model/ --restore_file xxx.pth
```
## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@InProceedings{jiang_2023_aaai,
    author  = {Jiang, Hai and Li, Haipeng and Lu, Yuhang and Han, Songchen and Liu, Shuaicheng},
    title = {Semi-supervised Deep Large-baseline Homography Estimation with Progressive Equivalence Constraint}},
    booktitle = {Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI)}
    year = {2023}
}
```
