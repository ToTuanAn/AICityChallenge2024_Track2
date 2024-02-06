export LOG_DIR=./logs
export DATA_DIR=./datasets
export TASK_NAME=ai_city_challenge
export MODEL=caption_style_transfer
export TEXT_COLUMN=sentence
export SUMMARY_COLUMN=gt
export DATA_RATIO=1
export DATASET_NAME=ai_city_challenge
export NLPEXP=./output
export MAX_TARGET_LENGTH=320
mkdir -p $DATA_DIR/$TASK_NAME/$DATA_RATIO/$MODEL
mkdir -p $LOG_DIR/$TASK_NAME/$DATA_RATIO/
touch $LOG_DIR/$TASK_NAME/$DATA_RATIO/$MODEL
python run.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --train_file $DATA_DIR/BART_denoising_data/train.csv \
    --validation_file $DATA_DIR/BART_denoising_data/val.csv \
    --source_prefix "rephrase: " \
    --output_dir $DATA_DIR/$TASK_NAME/$DATA_RATIO/$MODEL \
    --overwrite_output_dir \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --predict_with_generate \
    --max_target_length=$MAX_TARGET_LENGTH
