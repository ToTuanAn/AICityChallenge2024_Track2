# [CVPRW] 2024 AI City Challenge: Traffic Safety Description and Analysis

- Paper: To be update later 
- Contributior: Tuan-An To, Minh-Nam Tran, Trong-Bao Ho, Thien-Loc Ha, Quang-Tan Nguyen, Hoang-Chau Luong, Thanh-Duy Cao

## Framework

## Setup 
```
virtualenv venv
scoure venv/bin/activate
pip install -r requirements.txt
``` 

## Single-view 
### Preprocessing
```
cd vid2seq
# Construct a csv file that each row is a <video_path>, <feature_path>. For example: data/video1.mp4, data/video1.npy
python extract/make_csv.py
python extract/extract.py --csv <csv_path> --framerate 30
``` 

### Inference
1. Go to notebooks/vid2seq_inference.ipynb and construct the checkpoint path and test-set embedded feature path. Then run the notebook.
2. Go to scripts/get_submission.sh and construct the checkpoint path and test-set embedded feature path
```
cd vid2seq
bash ../scripts/get_submission.sh
```

### Training 
Upload the train-set embedded feature on Kaggle
Training notebook on Kaggle: [[TRAINING] SINGLE VIEW MODEL](https://www.kaggle.com/code/anttun/training-single-view-model/edit)

## Motion-Blur
Please check [Motion-Blur Link](https://github.com/ToTuanAn/AICityChallenge2024_Track2/blob/main/motion_blur/README.md)!

## Multi-view

## Acknowledgement
