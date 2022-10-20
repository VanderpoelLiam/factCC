#! /bin/bash
# Evaluate FactCC model

# UPDATE PATHS BEFORE RUNNING SCRIPT
export CODE_PATH=/home/liam/factCC/modeling
export DATA_PATH=/home/liam/factCC/data/bart_0.0000E+00_0.0000E+00
export CKPT_PATH=/home/liam/factCC/factcc-checkpoint

export TASK_NAME=factcc_annotated
export MODEL_NAME=bert-base-uncased

python3 $CODE_PATH/run.py \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_all_checkpoints \
  --do_lower_case \
  --overwrite_cache \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 12 \
  --model_type bert \
  --model_name_or_path $MODEL_NAME \
  --data_dir $DATA_PATH \
  --output_dir $CKPT_PATH
