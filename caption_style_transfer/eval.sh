export LOG_DIR=./logs
export DATA_DIR=./datasets
export TASK_NAME=ai_city_challenge
export MODEL=caption_style_transfer
export TEXT_COLUMN=sentence
export DATA_RATIO=1
export SUMMARY_COLUMN=gt
export MAX_TARGET_LENGTH=320
python eval.py \
    --model_name_or_path $DATA_DIR/$TASK_NAME/$DATA_RATIO/$MODEL \
    --do_train \
    --do_eval \
    --train_file $DATA_DIR/BART_denoising_data/train.csv \
    --validation_file $DATA_DIR/BART_denoising_data/val.csv \
    --source_prefix "paraphase: " \
    --output_dir $DATA_DIR/$TASK_NAME/$DATA_RATIO/$MODEL \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --max_target_length=$MAX_TARGET_LENGTH
