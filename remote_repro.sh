#!/usr/bin/env bash
# Runs ON the Lambda GPU instance, DETACHED (survives local disconnect/sleep).
# All progress is written to the NFS project outputs/ so results + diagnostics
# persist even if the instance is later terminated (watchdog / manual).
#
# Diagnostics for "what killed the run":
#   outputs/heartbeat.log  - wall-clock UTC pulse every 60s (last line = when the
#                            REMOTE stopped; cross-ref with local pmset sleep log
#                            and the watchdog fire time to attribute the cause).
#   outputs/status.txt     - phase timestamps (start / venv / data / train / DONE).
#   outputs/train_ml100k.log - full training/eval stdout (per-epoch, timestamped
#                            by main.py logging).
#   outputs/DONE           - written only on normal completion; contains the rc.
set -uo pipefail   # deliberately NO -e: we must always reach the DONE marker.
export TOKENIZERS_PARALLELISM=false

EPOCHS="${EPOCHS:-20}"; SAMPLE="${SAMPLE:-3,3}"; TEST_EPOCH="${TEST_EPOCH:-5}"
mkdir -p outputs checkpoints log model
ts() { date -u +%FT%TZ; }
say() { echo "$(ts) $*" | tee -a outputs/status.txt; }

rm -f outputs/DONE
say "job start host=$(hostname) epochs=$EPOCHS sample=$SAMPLE test_epoch=$TEST_EPOCH"

# wall-clock heartbeat to NFS (persists past instance death)
( while true; do echo "$(ts) heartbeat"; sleep 60; done >> outputs/heartbeat.log ) &
HB=$!
trap 'kill $HB 2>/dev/null' EXIT

say "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ' ')"

say "building venv on LOCAL disk ($HOME)"
VENV="$HOME/openp5-venv"
python3 -m venv "$VENV"; source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q "torch==2.2.2" "transformers==4.26.0" "numpy<2" "scikit-learn" \
  "scipy" "sentencepiece" "protobuf<3.21" "tqdm"
say "torch: $(python -c 'import torch;print(torch.__version__,"cuda",torch.cuda.is_available())' 2>&1)"

say "build ML100K data"
if [ ! -f data/ML100K/user_sequence.txt ]; then
  mkdir -p raw_data && ( cd raw_data && \
    curl -sSL -o ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip && \
    unzip -q -o ml-100k.zip )
  python raw_data/build_user_sequence.py
fi

say "train start (sample=$SAMPLE epochs=$EPOCHS)"
python -u src/src_t5/main.py \
  --datasets ML100K --data_path ./data --prompt_file ./prompt.txt \
  --model_dir ./model --log_dir ./log --checkpoint_dir ./checkpoint \
  --distributed 0 --gpu 0 --backbone t5-small \
  --tasks sequential,straightforward --item_indexing sequential \
  --epochs "$EPOCHS" --batch_size 64 --eval_batch_size 32 \
  --sample_prompt 1 --sample_num "$SAMPLE" --max_his 20 \
  --lr 1e-3 --train 1 --test_prompt seen:0 --test_before_train 0 \
  --test_epoch "$TEST_EPOCH" --valid_select 0 --random_initialize 1 \
  > outputs/train_ml100k.log 2>&1
RC=$?

say "train exited rc=$RC ; staging results"
cp -f log/ML100K/*.log outputs/ 2>/dev/null || true
grep -nE "hit@|ndcg@|testing .*dataset on|average training loss" \
  outputs/train_ml100k.log > outputs/metrics.txt 2>/dev/null || true
cp -f model/ML100K/*.pt checkpoints/ 2>/dev/null || true
say "DONE rc=$RC"
echo "$RC" > outputs/DONE
