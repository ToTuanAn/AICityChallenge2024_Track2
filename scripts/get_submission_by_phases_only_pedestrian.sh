mkdir output
pedestrian_pred_path=output/pedestrian.json
vehicle_pred_path=output/vehicle.json
submission_path=output/submission.json

mkdir data
cp -r $pedestrian_data_path data

python -m torch.distributed.launch --nproc_per_node 1 \
                                    --use_env \
                                    infer_by_phases.py \
                                    --combine_datasets wts \
                                    --model_name google-t5/t5-large \
                                    --batch_size_val 1 \
                                    --max_input_tokens 256 \
                                    --max_output_tokens 256 \
                                    --wts_test_json_path data/wts/pedestrian_test.json \
                                    --rule_mode pedestrian \
                                    --rule_config_path rules_engine/configs/rule_config.yaml \
                                    --load_internal_ckpt_phase_0 $pedestrian_ckpt_path_0 \
                                    --load_internal_ckpt_phase_1 $pedestrian_ckpt_path_1 \
                                    --load_internal_ckpt_phase_2 $pedestrian_ckpt_path_2 \
                                    --load_internal_ckpt_phase_3 $pedestrian_ckpt_path_3 \
                                    --load_internal_ckpt_phase_4 $pedestrian_ckpt_path_4 \
                                    --save $pedestrian_pred_path \
                                    --save_phase_0 $pedestrian_pred_path_0 \
                                    --save_phase_1 $pedestrian_pred_path_1 \
                                    --save_phase_2 $pedestrian_pred_path_2 \
                                    --save_phase_3 $pedestrian_pred_path_3 \
                                    --save_phase_4 $pedestrian_pred_path_4 
                                    
rm -r data

