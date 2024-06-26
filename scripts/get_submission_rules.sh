mkdir output
pedestrian_pred_path=output/pedestrian.json
vehicle_pred_path=output/vehicle.json
submission_path=output/submission.json

mkdir data
cp -r $pedestrian_data_path data
cp -r $pedestrian_external_data data

python -m torch.distributed.launch --nproc_per_node 1 \
                                    --use_env \
                                    infer.py \
                                    --combine_datasets wts \
                                    --model_name google-t5/t5-large \
                                    --batch_size_val 1 \
                                    --max_input_tokens 256 \
                                    --max_output_tokens 256 \
                                    --wts_test_json_path data/wts/pedestrian_test.json \
                                    --rule_mode pedestrian \
                                    --rule_config_path rules_engine/configs/rule_config.yaml \
                                    --load $pedestrian_ckpt_path \
                                    --save $pedestrian_pred_path 
                                    
rm -r data

mkdir data
cp -r $vehicle_data_path data
cp -r $vehicle_external_data data

python -m torch.distributed.launch --nproc_per_node 1 \
                                    --use_env \
                                    infer.py \
                                    --combine_datasets wts \
                                    --model_name google-t5/t5-large \
                                    --batch_size_val 1 \
                                    --max_input_tokens 256 \
                                    --max_output_tokens 256 \
                                    --wts_test_json_path data/wts/vehicle_test.json \
                                    --rule_mode vehicle \
                                    --rule_config_path rules_engine/configs/rule_config.yaml \
                                    --load $vehicle_ckpt_path \
                                    --save $vehicle_pred_path 

python postprocessing.py --pedestrian $pedestrian_pred_path \
                         --vehicle $vehicle_pred_path \
                         --save $submission_path \
                         --combine_datasets wts \