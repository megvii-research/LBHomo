
# [AAAI 2023] Semi-supervised Deep Large-baseline Homography Estimation with Progressive Equivalence Constraint. [[Paper]](https://arxiv.org/abs/2212.02763).
## Pipeline
![](https://github.com/megvii-research/LBHomo/blob/main/Figs/Pipeline.jpg)
## Dependencies
```
pip install -r requirements.txt
````
## Download the raw dataset
Please refer to [Content-Aware Unsupervised Deep Homography Estimation.](https://github.com/JirongZhang/DeepHomography).

- Dataset download links: [GoogleDriver](https://drive.google.com/file/d/19d2ylBUPcMQBb_MNBBGl9rCAS7SU-oGm/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/1Dkmz4MEzMtBx-T7nG0ORqA) (key: gvor)

- Unzip the data to directory "./dataset"

- Run "video2img.py"
```
Be sure to scale the image to (640, 360) since the point coordinate system is based on the (640, 360).e.g. img = cv2.imresize(img, (640, 360))
```
- Using the images in "train.txt" and "test.txt" for training and evaluation, the manually labeled evaluation files can be download from: [GoogleDriver], [BaiduYun].
## Pre-trained model
```
The models provided below are the retrained version(with minor differences in quantitative results)
```

| model    | RE | LT | LL | LF | SF | Avg | Model |
| --------- | ----------- | ------------ |------------ |------------ |------------ |------------ |------------ |
| Pre-trained | 1.66 | 5.49 | 4.11 | 7.57 | 6.95  | 5.16  |Baidu [Google] 
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
