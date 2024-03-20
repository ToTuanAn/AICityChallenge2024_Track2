# srun -p video -N1 -n8 --gres=gpu:8 --cpus-per-task=12 --quotatype=reserved --job-name=train_lsmdc \
python -u -m main_task_retrieval  \
    --do_train \
    --num_thread_reader=8 \
    --n_display=50 \
    --epochs=5 \
    --lr=1e-3 \
    --coef_lr=2e-3 \
    --batch_size=64 \
    --batch_size_val=16 \
    --features_path="" \
    --data_path="./data/LSMDC/" \
    --datatype="lsmdc" \
    --max_words=77 \
    --max_frames=8 \
    --feature_framerate=1 \
    --pretrained_clip_name="ViT-L/14" \
    --slice_framepos=2 \
    --loose_type \
    --linear_patch=2d \
    --sim_header=meanP \
    --output_dir="" \
    --freeze_layer_num=0 \
    --expand_msrvtt_sentences \
    --mergeclip=True \
    --clip_evl \
    --pretrained_path="" \