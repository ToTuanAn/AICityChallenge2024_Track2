# [CVPRW] 2024 AI City Challenge: Traffic Safety Description and Analysis

- Paper: [To be updated] 
- Contributior: Tuan-An To, Minh-Nam Tran, Trong-Bao Ho, Thien-Loc Ha, Quang-Tan Nguyen, Hoang-Chau Luong, Thanh-Duy Cao

## Framework
![Alt text](assets/OverviewMethod.png?raw=true)

## Setup 
```
virtualenv venv
scoure venv/bin/activate
pip install -r requirements.txt
``` 

## Single-view 
### Preprocessing
1. Convert to from the WTS to YouCook format.

Convert the competition dataset to YouCook-format annotation files by running the following commands:
```bash
python bdd5k2youcook_by_phase.py --videos_dir datasets/BDD_PC_5K/videos --caption_dir datasets/BDD_PC_5K/captions --output_dir annotations/BDD_PC_5K --merge_val

python wts2youcook_by_phase.py --videos_dir datasets/BDD_PC_5K/videos --caption_dir datasets/WTS/captions --output_dir annotations/WTS --merge_val
```

where the `--merge_val` flag is used to merge the validation set into the training set, `--videos_dir` is the path to the video directory, `--caption_dir` is the path to the caption directory, and `--output_dir` is the path to the output directory.

The structure of the video directory of BDD_PC_5K should be as follows:
```
bdd_pc_5k_video_dir /
├── train /
│   └── vehicle_view /
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
│   
├── val /
│   └── vehicle_view /
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
```

and the structure of the video directory of WTS should be as follows:
```
wts_video_dir /
├── train /
│   ├── video_id_1 /
│   │   ├── overhead_view /
│   │   |   ├── abc.mp4
│   │   |   └── ...
│   │   └── vehicle_view /
│   │       ├── abc.mp4
│   │       └── ...
│   ├── video_id_2 /
│   │   └── overhead_view /
│   │       ├── abc.mp4
│   │       └── ...
│   └── ...
│
├── val /
│   ├── video_id_1 /
│   |   ├── overhead_view /
│   |   |   ├── abc.mp4
│   |   |   └── ...
│   |   └── vehicle_view /
│   |       ├── abc.mp4
│   |       └── ...
│   ├── video_id_2 /
│   └── ...
```

The caption directory structures are similar to the corresponding video directory structures.


2. Construct a csv file that each row is a <video_path>, <feature_path>. For example: data/video1.mp4, data/video1.npy
```
cd vid2seq
python extract/make_csv.py
python extract/extract.py --csv <csv_path> --framerate 30
``` 


### Inference
1. Go to notebooks/vid2seq_inference.ipynb and construct the checkpoint path and test-set embedded feature path. Then run the notebook.
2. Go to scripts/get_submission.sh and construct the checkpoint path and test-set embedded feature path.
```
cd vid2seq
bash ../scripts/get_submission.sh
```

### Training 
1. Upload the train-set embedded feature on Kaggle.
2. Training notebook on Kaggle: [[TRAINING] SINGLE VIEW MODEL](https://www.kaggle.com/code/anttun/training-single-view-model/edit).

## Motion-Blur
Please check [Motion-Blur Link](https://github.com/ToTuanAn/AICityChallenge2024_Track2/blob/main/motion_blur/README.md).

## Multi-view
[To be updated]

## Acknowledgement
We would like to thank the [Vid2Seq](https://github.com/antoyang/VidChapters) repository for their outstanding video captioning.