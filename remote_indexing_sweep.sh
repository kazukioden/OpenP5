#!/usr/bin/env bash
# Runs ON the Lambda GPU instance, DETACHED. Sweeps item-indexing methods on
# ML100K with an otherwise-identical config, writing each to outputs/<indexing>/
# so the three runs never collide (safe to also run one-per-instance in parallel:
# pass a single method via INDEXINGS).
#
#   INDEXINGS  space-separated subset of: random sequential collaborative
#   EPOCHS SAMPLE BATCH VALID_SELECT  training config (paper-matched defaults)
#
# Diagnostics per method: outputs/<m>/{status.txt,heartbeat.log,train.log,metrics.txt,DONE}
set -uo pipefail
export TOKENIZERS_PARALLELISM=false

INDEXINGS="${INDEXINGS:-random sequential collaborative}"
EPOCHS="${EPOCHS:-12}"; SAMPLE="${SAMPLE:-3,3}"; BATCH="${BATCH:-128}"
VALID_SELECT="${VALID_SELECT:-1}"          # 1 = report best-by-val model (no test peeking)
CTS="${CTS:-100}"; CCLUSTER="${CCLUSTER:-10}"   # collaborative: paper's ts=100, cluster=10
ts() { date -u +%FT%TZ; }

mkdir -p outputs
( while true; do echo "$(ts) heartbeat [$INDEXINGS]"; sleep 60; done >> outputs/heartbeat.log ) &
HB=$!; trap 'kill $HB 2>/dev/null' EXIT

echo "$(ts) sweep start: methods=[$INDEXINGS] epochs=$EPOCHS sample=$SAMPLE batch=$BATCH valid_select=$VALID_SELECT" | tee outputs/sweep_status.txt

# one-time env on LOCAL disk (shared across methods)
VENV="$HOME/openp5-venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"; source "$VENV/bin/activate"
  pip install -q --upgrade pip
  pip install -q "torch==2.2.2" "transformers==4.26.0" "numpy<2" "scikit-learn" \
    "scipy" "sentencepiece" "protobuf<3.21" "tqdm"
else
  source "$VENV/bin/activate"
fi
echo "$(ts) torch $(python -c 'import torch;print(torch.__version__,torch.cuda.is_available())')" | tee -a outputs/sweep_status.txt

# one-time data
if [ ! -f data/ML100K/user_sequence.txt ]; then
  mkdir -p raw_data && ( cd raw_data && \
    curl -sSL -o ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip && \
    unzip -q -o ml-100k.zip )
  python raw_data/build_user_sequence.py
fi

for M in $INDEXINGS; do
  O="outputs/$M"; mkdir -p "$O" "checkpoints/$M" "log_$M" "model_$M"
  rm -f "$O/DONE"
  echo "$(ts) === $M : train start ===" | tee -a outputs/sweep_status.txt "$O/status.txt"
  python -u src/src_t5/main.py \
    --datasets ML100K --data_path ./data --prompt_file ./prompt.txt \
    --model_dir "./model_$M" --log_dir "./log_$M" --checkpoint_dir "./checkpoints/$M" \
    --distributed 0 --gpu 0 --backbone t5-small \
    --tasks sequential,straightforward --item_indexing "$M" \
    --collaborative_token_size "$CTS" --collaborative_cluster "$CCLUSTER" \
    --epochs "$EPOCHS" --batch_size "$BATCH" --eval_batch_size 32 \
    --sample_prompt 1 --sample_num "$SAMPLE" --max_his 20 \
    --lr 1e-3 --train 1 --test_prompt seen:0 --test_before_train 0 --test_epoch 0 \
    --valid_select "$VALID_SELECT" --random_initialize 1 \
    > "$O/train.log" 2>&1
  RC=$?
  grep -nE "hit@|ndcg@|testing .*dataset on|valid loss|best validation" "$O/train.log" > "$O/metrics.txt" 2>/dev/null || true
  echo "$(ts) === $M : done rc=$RC ===" | tee -a outputs/sweep_status.txt "$O/status.txt"
  echo "$RC" > "$O/DONE"
done
echo "$(ts) SWEEP DONE" | tee -a outputs/sweep_status.txt
echo 0 > outputs/DONE
