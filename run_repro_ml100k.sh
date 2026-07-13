#!/bin/bash
# Reproduce the Wantedly blog: ML100K sequential recommendation, T5.
# Usage: ./run_repro_ml100k.sh <epochs> <sample_num> <test_epoch>
#   e.g. ./run_repro_ml100k.sh 10 1,1 5   (M3-reduced)
#        ./run_repro_ml100k.sh 20 3,3 5   (faithful, ~40h on M3)
cd /Users/kazuki_shimizu/recommend/generative_recommend/OpenP5
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
EPOCHS=${1:-10}
SAMPLE=${2:-1,1}
TEST_EPOCH=${3:-5}
.venv-p5/bin/python -u src/src_t5/main.py \
  --datasets ML100K --data_path ./data --prompt_file ./prompt.txt \
  --model_dir ./model --log_dir ./log --checkpoint_dir ./checkpoint \
  --distributed 0 --gpu 0 --backbone t5-small \
  --tasks sequential,straightforward --item_indexing sequential \
  --epochs "$EPOCHS" --batch_size 64 --eval_batch_size 32 \
  --sample_prompt 1 --sample_num "$SAMPLE" --max_his 20 \
  --lr 1e-3 --train 1 --test_prompt seen:0 --test_before_train 0 \
  --test_epoch "$TEST_EPOCH" --valid_select 0 --random_initialize 1
