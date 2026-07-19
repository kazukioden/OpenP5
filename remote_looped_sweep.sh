#!/usr/bin/env bash
# Runs ON the Lambda GPU instance, DETACHED. Sweeps INDEXINGS x LOOP_KS on one
# dataset, identical config otherwise. For 探求②: does decoder looping (compute)
# close the random↔collaborative (representation) gap? Per-run outputs are
# isolated at outputs/<dataset>_<indexing>_k<loop>/ so runs never collide
# (also fine to run a subset per instance in parallel).
#
#   DATASET     one of: ML100K LastFM        (default: LastFM)
#   INDEXINGS   subset of: random sequential collaborative   (default: random collaborative)
#   LOOP_KS     space-separated ints         (default: 1 2 4)
#   EPOCHS SAMPLE BATCH VALID_SELECT          (paper-matched defaults)
set -uo pipefail
export TOKENIZERS_PARALLELISM=false

DATASET="${DATASET:-LastFM}"
INDEXINGS="${INDEXINGS:-random collaborative}"
LOOP_KS="${LOOP_KS:-1 2 4}"
EPOCHS="${EPOCHS:-12}"; SAMPLE="${SAMPLE:-3,3}"; BATCH="${BATCH:-128}"
VALID_SELECT="${VALID_SELECT:-1}"; CTS="${CTS:-100}"; CCLUSTER="${CCLUSTER:-10}"
ts() { date -u +%FT%TZ; }

mkdir -p outputs
( while true; do echo "$(ts) heartbeat"; sleep 60; done >> outputs/heartbeat.log ) &
HB=$!; trap 'kill $HB 2>/dev/null' EXIT
echo "$(ts) looped sweep: data=$DATASET idx=[$INDEXINGS] loops=[$LOOP_KS] ep=$EPOCHS s=$SAMPLE b=$BATCH" | tee outputs/sweep_status.txt

VENV="$HOME/openp5-venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"; source "$VENV/bin/activate"
  pip install -q --upgrade pip
  pip install -q "torch==2.2.2" "transformers==4.26.0" "numpy<2" "scikit-learn" \
    "scipy" "sentencepiece" "protobuf<3.21" "tqdm"
else source "$VENV/bin/activate"; fi

# data
if [ "$DATASET" = "ML100K" ] && [ ! -f data/ML100K/user_sequence.txt ]; then
  mkdir -p raw_data && ( cd raw_data && curl -sSL -o ml-100k.zip \
    https://files.grouplens.org/datasets/movielens/ml-100k.zip && unzip -q -o ml-100k.zip )
  python raw_data/build_user_sequence.py
fi
if [ "$DATASET" = "LastFM" ] && [ ! -f data/LastFM/user_sequence.txt ]; then
  mkdir -p raw_data && ( cd raw_data && curl -sSL -o lastfm.zip \
    https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip && \
    unzip -q -o lastfm.zip -d lastfm )
  python raw_data/build_lastfm_sequence.py
fi

for M in $INDEXINGS; do
  for K in $LOOP_KS; do
    tag="${DATASET}_${M}_k${K}"; O="outputs/$tag"
    mkdir -p "$O" "checkpoints/$tag" "log_$tag" "model_$tag"; rm -f "$O/DONE"
    echo "$(ts) === $tag : start ===" | tee -a outputs/sweep_status.txt "$O/status.txt"
    python -u src/src_t5/main.py \
      --datasets "$DATASET" --data_path ./data --prompt_file ./prompt.txt \
      --model_dir "./model_$tag" --log_dir "./log_$tag" --checkpoint_dir "./checkpoints/$tag" \
      --distributed 0 --gpu 0 --backbone t5-small \
      --tasks sequential,straightforward --item_indexing "$M" --loop_k "$K" \
      --collaborative_token_size "$CTS" --collaborative_cluster "$CCLUSTER" \
      --epochs "$EPOCHS" --batch_size "$BATCH" --eval_batch_size 32 \
      --sample_prompt 1 --sample_num "$SAMPLE" --max_his 20 \
      --lr 1e-3 --train 1 --test_prompt seen:0 --test_before_train 0 --test_epoch 0 \
      --valid_select "$VALID_SELECT" --random_initialize 1 \
      > "$O/train.log" 2>&1
    RC=$?
    grep -nE "hit@|ndcg@|testing .*dataset on|best validation" "$O/train.log" > "$O/metrics.txt" 2>/dev/null || true
    echo "$(ts) === $tag : done rc=$RC ===" | tee -a outputs/sweep_status.txt "$O/status.txt"
    echo "$RC" > "$O/DONE"
  done
done
echo "$(ts) SWEEP DONE" | tee -a outputs/sweep_status.txt
echo 0 > outputs/DONE
