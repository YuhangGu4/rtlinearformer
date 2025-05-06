# rtlinearformer
RTLinearFormer: Semantic segmentation with lightweight linear attentions

This is the official repository for our recent work.

## Models
For simple reproduction, we provide the ImageNet pretrained models here.

| Model (ImageNet) | RTLinearFormer |
|:-:|:-:|
| Link | [download](https://drive.google.com/file/d/1hIBp_8maRr60-B3PF0NVtaA6TYBvO4y-/view?usp=sharing) |

Also, the finetuned models on Cityscapes and Camvid are available for direct application in road scene parsing.

| Model (Cityscapes) | Val (% mIOU) | FPS |
|:-:|:-:|:-:|
| RTLinearFormer | [78.41] | 66.7 |

| Model (CamVid) | Test (% mIOU)| FPS |
|:-:|:-:|:-:|
| RTLinearFormer | [77.4] | 143.2 |

## Prerequisites
This implementation is based on [PIDNet](https://github.com/XuJiacong/PIDNet). Please refer to their repository for installation and dataset preparation. The inference speed is tested on single RTX 3090 using the method introduced by [SwiftNet](https://arxiv.org/pdf/1903.08469.pdf). No third-party acceleration lib is used, so you can try [TensorRT](https://github.com/NVIDIA/TensorRT) or other approaches for faster speed.

## Usage

### 0. Prepare the dataset

* Download the [Cityscapes](https://www.cityscapes-dataset.com/) and [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) datasets and unzip them in `data/cityscapes` and `data/camvid` dirs.
* Check if the paths contained in lists of `data/list` are correct for dataset images.

### 1. Training

* Download the ImageNet pretrained models and put them into `pretrained_models/imagenet/` dir.
* For example, train the RTLinearFormer on Cityscapes with batch size of 3 on 1 GPU:
````bash
python tools/train.py --cfg configs/cityscapes/rtlinearformer_base_cityscapes.yaml GPUS (0) TRAIN.BATCH_SIZE_PER_GPU 3
````

### 2. Evaluation

* Download the finetuned models for Cityscapes and CamVid and put them into `pretrained_models/cityscapes/` and `pretrained_models/camvid/` dirs, respectively.
* For example, evaluate the RTLinearFormer on Cityscapes val set:
````bash
python tools/eval.py --cfg configs/cityscapes/rtlinearformer_base_cityscapes.yaml \
                          TEST.MODEL_FILE pretrained_models/cityscapes/rtlinearformer_base_Cityscapes_val.pt
````
* Or, evaluate the RTLinearFormer on CamVid test set:
````bash
python tools/eval.py --cfg configs/camvid/rtlinearformer_base_camvid.yaml \
                          TEST.MODEL_FILE pretrained_models/camvid/rtlinearformer_base_Camvid_Test.pt \
                          DATASET.TEST_SET list/camvid/test.lst
````
* Generate the testing results of RTLinearFormer on Cityscapes test set:
````bash
python tools/eval.py --cfg configs/cityscapes/rtlinearformer_base_cityscapes_trainval.yaml \
                          TEST.MODEL_FILE pretrained_models/cityscapes/rtlinearformer_base_Cityscapes_test.pt \
                          DATASET.TEST_SET list/cityscapes/test.lst
````

### 3. Speed Measurement

* Measure the inference speed of RTLinearFormer for Cityscapes:
````bash
python models/speed/rt_linear_transformer_speed_v5.py --a 'rtlinearformer' --c 19 --r 1024 2048
````

## Citation

If you think this implementation is useful for your work, please cite our paper:
```
@article{gu2025rtlinearformer,
  title={RTLinearFormer: Semantic segmentation with lightweight linear attentions},
  author={Gu, Yuhang and Fu, Chong and Song, Wei and Wang, Xingwei and Chen, Junxin},
  journal={Neurocomputing},
  pages={129489},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgement

* Our implementation is modified based on [PIDNet-Semantic-Segmentation](https://github.com/XuJiacong/PIDNet).
* Latency measurement code is borrowed from the [DDRNet](https://github.com/ydhongHIT/DDRNet).
* Thanks for their nice contribution.
