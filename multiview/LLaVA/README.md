## Multi-view
### Preprocessing

Suppose the directory structure of the WTS is as follows:
```
WTS_dataset /
├── annotations /
│   ├── caption /
│   │   ├── train /
│   │   |   ├── video_id_1 / 
│   │   |   |   ├── overhead_view /
|   |   |   |   |   └── video_id_1_caption.json
│   │   |   |   └── vehicle_view /
|   |   |   |       └── video_id_1_caption.json
|   |   |   ├── video_id_2 /
|   |   |   └── ...
|   |   ├── val /
|   |   |   └── ...
│   ├── bbox_annotated /
│   ├── bbox_generated /
|   └── ...
├── videos /
│   ├── video_id_1 /
│   |   ├── overhead_view /
│   |   |   ├── camera_view_1.mp4
|   |   |   ├── camera_view_2.mp4
│   |   |   └── ...
│   |   └── vehicle_view /
│   |       └── vehicle_view.mp4
│   ├── video_id_2 /
│   |   └── ...
|   ├── ...
```

Suppose the directory structure of the BDD_PC_5K is as follows:
```
BDD_PC_5K_dataset /
├── annotations /
│   ├── caption /
│   │   ├── train /
│   │   |   ├── video1_caption.json 
|   |   |   ├── video2_caption.json
|   |   |   └── ...
|   |   ├── val /
|   |   |   └── ...   
│   ├── bbox_annotated /
│   ├── bbox_generated /
|   └── ...
├── videos /
|   ├── train /
│   |   └── vehicle_view /
│   |        ├── video1.mp4
│   |        ├── video2.mp4
│   |        └── ... 
|   └── val /
│       └── vehicle_view /
│           ├── video1.mp4
│           ├── video2.mp4
│           └── ...
```

Then run the following commands to convert the current dataset's format to LLaVA's format and trim videos into multiple event phases.
```bash
python preprocess_to_llava_wts.py --annotation-folder WTS_dataset/annotations --video_folder WTS_dataset/annotations --annotation-output-folder your_output_folder/annotations --video-output-folder your_output_folder/videos
python preprocess_bdd_pc_5k.py --annotation-folder BDD_PC_5K_dataset/annotations --video_folder BDD_PC_5K_dataset/annotations --annotation-output-folder your_output_folder/annotations --video-output-folder your_output_folder/videos
```

For each WTS and BDD_PC_5K dataset, the resulting preprocessed directory structure is as follows, where each json file has the corresponding LLaVA's format:
```
your_output_folder /
├── instructs /
|   ├── pedestrian / 
│   |   ├── video_id_1.json
|   |   ├── video_id_2.json
│   │   └── ...
|   ├── vehicle /
|   |   ├── video_id_1.json
|   |   ├── video_id_2.json
│   │   └── ...
├── videos /
│   ├── video_id_1 /
│   |   ├── phase_0 /
│   |   |   ├── camera_view_1.mp4
|   |   |   ├── camera_view_2.mp4
|   |   |   ├── vehicle_view.mp4
│   |   |   └── ...
│   |   ├── phase_1 /
│   |   |   ├── camera_view_1.mp4
|   |   |   ├── camera_view_2.mp4
|   |   |   ├── vehicle_view.mp4
│   |   |   └── ...
|   |   └── ...
│   ├── video_id_2 /
│   └── ...
```

### Training
1. Merge all ```video_id_x``` folders in the ```your_output_folder/videos``` folder of the WTS and BDD_PC_5K dataset into one ```videos``` folder, then upload it onto Kaggle.
2. Merge all ```video_id_x.json``` files in the ```your_output_folder/instructs/pedestrian``` folder of the WTS and BDD_PC_5K dataset into one ```pedestrian``` folder, then upload it onto Kaggle.
3. Merge all ```video_id_x.json``` files in the ```your_output_folder/instructs/vehicle``` folder of the WTS and BDD_PC_5K dataset into one ```vehicle``` folder, then upload it onto Kaggle.
4. Upload the [TRAINING NOTEBOOK](https://github.com/ToTuanAn/AICityChallenge2024_Track2/tree/main/multiview/LLaVA/notebooks/train.ipynb) onto Kaggle and add the uploaded dataset at step 1, 2, 3.
5. Adjust some arguments below to fit with your case. Download the [LINEAR PROJECTOR](https://huggingface.co/LanguageBind/Video-LLaVA-Pretrain-7B/resolve/main/mm_projector.bin?download=true).
```
--data_path: the path to the folder containing the instruct json files
--video_folder: the path to the video folder
--output_dir: the path to the folder for saving checkpoints, the default value is set to /kaggle/working/multiview-videollava
--pretrain_mm_mlp_adapter: the path to the LINEAR PROJECTOR
```
6. Once the training is completed, go into the ```output_dir``` folder, save all files except the ```checkpoint-x``` folder.
7. Open the previously saved ```config.json``` file, delete the ```"quantization_config"``` key and value pair.
8. Upload all saved files onto HuggingFace. 

### Inference
1. Preprocess the test dataset as correspoding to the train dataset
2. Inference notebook on Kaggle: [INFERENCE NOTEBOOK](https://github.com/ToTuanAn/AICityChallenge2024_Track2/tree/main/multiview/LLaVA/notebooks/infer.ipynb)
3. Adjust some arguments below to fit with your case. 
```
--model-path: the HuggingFace repo's name of the trained model.
--offload-folder: the path to the folder for offloading the pretrained model's weights
--data-path: the path to the folder containing the instruct json files
--output-path: the json file path to save the inference results
--video-folder: the path to the video folder
```




