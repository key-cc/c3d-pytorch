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


### Training

To train a video recognition model with pre-trained image models (for Kinetics-400 and Kineticc-600 datasets), run:


Colaboratory file: [C3D_Hockey](c3d_hockey.ipynb) 

## Results and Models

### HockeyFights

| Model | Input size | acc |
| :---: | :---: | :---: | 
|  C3D  |     16 x 70 x 110     |  93.50  | 
|  R21D  |     8 x 112 x 112      |  96.00  |
|  P3D  |     16 x 160 x 160      |  98.00 |
|  ARTNet  |     16x112x112      |  98.00  |
