#!/usr/bin/env bash
# Runs ON the Lambda GPU instance (cwd = pushed OpenP5 project dir).
# Invoked via: lambda_run_once.sh "bash remote_repro.sh"
# Builds env + ML100K data, trains+evals the T5 sequential model on CUDA,
# and stages results into outputs/ + checkpoints/ for pull-back.
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

EPOCHS="${EPOCHS:-20}"
SAMPLE="${SAMPLE:-3,3}"
TEST_EPOCH="${TEST_EPOCH:-5}"

echo "== GPU check =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "== python env (pin transformers 4.26 for P5_T5) =="
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install -q --upgrade pip
# pip's default linux torch wheel is a CUDA build; matches the instance driver.
pip install -q "torch" "transformers==4.26.0" "numpy<2" "scikit-learn" \
  "scipy" "sentencepiece" "protobuf<3.21" "tqdm"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

echo "== build ML100K data (excluded from push) =="
if [ ! -f data/ML100K/user_sequence.txt ]; then
  mkdir -p raw_data && ( cd raw_data && \
    curl -sSL -o ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip && \
    unzip -q -o ml-100k.zip )
  python raw_data/build_user_sequence.py
fi

echo "== train + eval (faithful: sample ${SAMPLE}, ${EPOCHS} epochs) =="
mkdir -p outputs checkpoints log model
python -u src/src_t5/main.py \
  --datasets ML100K --data_path ./data --prompt_file ./prompt.txt \
  --model_dir ./model --log_dir ./log --checkpoint_dir ./checkpoint \
  --distributed 0 --gpu 0 --backbone t5-small \
  --tasks sequential,straightforward --item_indexing sequential \
  --epochs "$EPOCHS" --batch_size 64 --eval_batch_size 32 \
  --sample_prompt 1 --sample_num "$SAMPLE" --max_his 20 \
  --lr 1e-3 --train 1 --test_prompt seen:0 --test_before_train 0 \
  --test_epoch "$TEST_EPOCH" --valid_select 0 --random_initialize 1 \
  2>&1 | tee outputs/train_ml100k.log

echo "== stage results for pull =="
cp -f log/ML100K/*.log outputs/ 2>/dev/null || true
grep -nE "hit@|ndcg@|testing .*dataset on|average training loss" outputs/train_ml100k.log \
  > outputs/metrics.txt || true
cp -f model/ML100K/*.pt checkpoints/ 2>/dev/null || true
echo "DONE. metrics -> outputs/metrics.txt"
