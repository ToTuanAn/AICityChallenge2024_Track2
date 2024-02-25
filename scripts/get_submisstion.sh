pedestrain_ckpt_path=/kaggle/input/best-vehicle-model-ckpt/vehicle_39.93.pth
vehicle_ckpt_path=/kaggle/input/best-vehicle-model-ckpt/vehicle_39.93.pth

pedestrian_data_path=/kaggle/input/dummy-pedestrian/wts
vehicle_data_path=/kaggle/input/dummy-vehicle/wts

pedestrain_pred_path=output/pedestrian.json
vehicle_pred_path=output/vehicle.json

mkdir output

mkdir data
cp -r $pedestrian_data_path data

python -m torch.distributed.launch --nproc_per_node 1 \
                                    --use_env \
                                    infer.py \
                                    --combine_datasets wts \
                                    --model_name t5-large \
                                    --batch_size_val 1 \
                                    --max_input_tokens 256 \
                                    --max_output_tokens 256 \
                                    --load $pedestrian_ckpt_path \
                                    --save $pedestrian_pred_path

rm -r data
mkdir data
cp -r $vehicle_data_path data

python -m torch.distributed.launch --nproc_per_node 1 \
                                    --use_env \
                                    infer.py \
                                    --combine_datasets wts \
                                    --model_name t5-large \
                                    --batch_size_val 1 \
                                    --max_input_tokens 256 \
                                    --max_output_tokens 256 \
                                    --load $vehicle_ckpt_path \
                                    --save $vehicle_pred_path