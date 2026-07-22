#!/bin/bash
set -e
export WANDB_MODE=disabled
export DISABLE_VERSION_CHECK=1
cd /app
python src/train.py \
    --model_name_or_path /models/Qwen3-1.7B \
    --trust_remote_code true \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --dataset metamathqa \
    --cutoff_len 1024 \
    --max_samples 20000 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 4 \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED:-42} \
    --logging_steps 20 \
    --save_steps 99999 \
    --overwrite_output_dir true \
    --save_only_model true \
    --report_to none \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true
echo "Training complete. Model saved to ${OUTPUT_DIR}"
