#!/bin/bash
cd /Users/kazuki_shimizu/recommend/generative_recommend/OpenP5
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
.venv-p5/bin/python -u src/src_t5/main.py \
  --datasets ML100K_tiny --data_path ./data --prompt_file ./prompt.txt \
  --model_dir ./model --log_dir ./log --checkpoint_dir ./checkpoint \
  --distributed 0 --gpu 0 --backbone t5-small \
  --tasks sequential,straightforward --item_indexing sequential \
  --epochs 1 --batch_size 64 --eval_batch_size 32 \
  --sample_prompt 1 --sample_num 3,3 --max_his 20 \
  --lr 1e-3 --train 1 --test_prompt seen:0 --test_before_train 0 --test_epoch 0 \
  --valid_select 0 --random_initialize 1
