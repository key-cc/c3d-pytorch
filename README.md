# video action recognition

Video classification using Dataset: HockeyFights
```
c3d-pytorch
├── extract_frames.py
├── hockey_data.py
├── hockey_model.py
├── test.txt
├── train.txt
├── log.py
├── hockey_train.h5
├── hockey_test.h5
├── data
│   ├── fi100_xvid
│   ├── fi101_xvid
│   ├── fi102_xvid
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├──  ...
│   ├── ...
├── video
│   ├── HockeyFights
│   │   ├── fi100_xvid.avi
│   │   ├── fi101_xvid.avi
│   │   ├── fi102_xvid.avi
│   │   ├──  ...

```
## Setup
###  Data Preparation

Please refer to [data_preparation.md](docs/data_preparation.md) for a general knowledge of data preparation.
The supported datasets are listed in [supported_datasets.md](docs/supported_datasets.md).

We also share our Kinetics-400 annotation file [k400_val](https://github.com/SwinTransformer/storage/releases/download/v1.0.6/k400_val.txt), [k400_train](https://github.com/SwinTransformer/storage/releases/download/v1.0.6/k400_train.txt) for better comparison.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --eval top_k_accuracy

# multi-gpu testing
bash tools/dist_test.sh <CONFIG_FILE> <CHECKPOINT_FILE> <GPU_NUM> --eval top_k_accuracy
```

### Training

To train a video recognition model with pre-trained image models (for Kinetics-400 and Kineticc-600 datasets), run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.backbone.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
bash tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.backbone.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

Colaboratory file: [C3D_Hockey](c3d_hockey.ipynb) 

## Results and Models

### HockeyFights

| Model | Input size | acc | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  C3D  |     16 x 70 x 110     |  93.50  |   28M   |  87.9G  |  [config](configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py)  | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth)/[baidu](https://pan.baidu.com/s/1mIqRzk8RILeRsP2KB5T6fg) |
|  R21D  |     8 x 112 x 112      |  96.00  |   50M   |  165.9G  |  [config](configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py)   | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_small_patch244_window877_kinetics400_1k.pth)/[baidu](https://pan.baidu.com/s/1imq7LFNtSu3VkcRjd04D4Q) |
|  P3D  |     16 x 160 x 160      |  98.00 |   88M   |  281.6G  |  [config](configs/recognition/swin/swin_base_patch244_window877_kinetics400_1k.py)   | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_1k.pth)/[baidu](https://pan.baidu.com/s/1bD2lxGxqIV7xECr1n2slng) |
|  ARTNet  |     16x112x112      |  98.00  |   88M   |  281.6G  |  [config](configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py)   | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth)/[baidu](https://pan.baidu.com/s/1CcCNzJAIud4niNPcREbDbQ) |
