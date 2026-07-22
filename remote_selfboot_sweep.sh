#!/usr/bin/env bash
# Runs ON the Lambda GPU instance, DETACHED. Self-bootstrap (ReST) sweep:
# per indexing method, train base M0 (= "off"), then R rounds of
# gen_pseudo(frozen M_{r-1}) -> train M_r with real+pseudo (= "on").
# Baseline "off" also serves as the 探求① sanity check.
#
#   DATASET LastFM|ML100K   INDEXINGS "random collaborative"   ROUNDS 1
#   EPOCHS SAMPLE BATCH  PSEUDO_W GEN_TOPK PER_USER
set -uo pipefail
export TOKENIZERS_PARALLELISM=false

DATASET="${DATASET:-LastFM}"
INDEXINGS="${INDEXINGS:-random collaborative}"
ROUNDS="${ROUNDS:-1}"
EPOCHS="${EPOCHS:-12}"; SAMPLE="${SAMPLE:-3,3}"; BATCH="${BATCH:-128}"
PSEUDO_W="${PSEUDO_W:-0.3}"; GEN_TOPK="${GEN_TOPK:-5}"; PER_USER="${PER_USER:-1}"
CTS="${CTS:-100}"; CCLUSTER="${CCLUSTER:-10}"
ts(){ date -u +%FT%TZ; }

mkdir -p outputs ckpt pseudo
( while true; do echo "$(ts) heartbeat"; sleep 60; done >> outputs/heartbeat.log ) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
echo "$(ts) selfboot: data=$DATASET idx=[$INDEXINGS] rounds=$ROUNDS s=$SAMPLE b=$BATCH pw=$PSEUDO_W" | tee outputs/sweep_status.txt

VENV="$HOME/openp5-venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"; source "$VENV/bin/activate"; pip install -q --upgrade pip
  pip install -q "torch==2.2.2" "transformers==4.26.0" "numpy<2" "scikit-learn" "scipy" "sentencepiece" "protobuf<3.21" "tqdm"
else source "$VENV/bin/activate"; fi

if [ "$DATASET" = "LastFM" ] && [ ! -f data/LastFM/user_sequence.txt ]; then
  mkdir -p raw_data && ( cd raw_data && curl -sSL -o lastfm.zip https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip && unzip -q -o lastfm.zip -d lastfm ); python raw_data/build_lastfm_sequence.py
fi
if [ "$DATASET" = "ML100K" ] && [ ! -f data/ML100K/user_sequence.txt ]; then
  mkdir -p raw_data && ( cd raw_data && curl -sSL -o ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip && unzip -q -o ml-100k.zip ); python raw_data/build_user_sequence.py
fi

COMMON="--datasets $DATASET --data_path ./data --prompt_file ./prompt.txt --distributed 0 --gpu 0 --backbone t5-small --tasks sequential,straightforward --item_indexing IDX --collaborative_token_size $CTS --collaborative_cluster $CCLUSTER --sample_prompt 1 --sample_num $SAMPLE --max_his 20 --random_initialize 1"

train_round(){ # $1=method $2=tag $3=model_name $4=pseudo_file(optional)
  local M=$1 tag=$2 mn=$3 pf=${4:-} O="outputs/$2"
  mkdir -p "$O" "log_$tag" "model_$tag"; rm -f "$O/DONE"
  local extra=""; [ -n "$pf" ] && extra="--pseudo_file $pf --pseudo_weight $PSEUDO_W"
  echo "$(ts) === $tag : train ===" | tee -a outputs/sweep_status.txt "$O/status.txt"
  python -u src/src_t5/main.py ${COMMON/IDX/$M} \
    --model_dir "./model_$tag" --log_dir "./log_$tag" --checkpoint_dir ./ckpt --model_name "$mn" \
    --epochs "$EPOCHS" --batch_size "$BATCH" --eval_batch_size 32 --lr 1e-3 \
    --train 1 --test_prompt seen:0 --test_before_train 0 --test_epoch 0 --valid_select 1 $extra \
    > "$O/train.log" 2>&1
  local rc=$?
  grep -nE "hit@|ndcg@|testing .*dataset on" "$O/train.log" > "$O/metrics.txt" 2>/dev/null || true
  echo "$rc" > "$O/DONE"
}

for M in $INDEXINGS; do
  base="${DATASET}_${M}"
  train_round "$M" "${base}_off" "${base}_m0.pt"          # round 0 = baseline / sanity
  prev="ckpt/${base}_m0.pt"
  for r in $(seq 1 "$ROUNDS"); do
    echo "$(ts) === ${base} r${r} : gen_pseudo ===" | tee -a outputs/sweep_status.txt
    python -u src/src_t5/gen_pseudo.py ${COMMON/IDX/$M} \
      --load_path "$prev" --pseudo_out "pseudo/${base}_r${r}.txt" \
      --gen_topk "$GEN_TOPK" --pseudo_per_user "$PER_USER" > "outputs/${base}_gen_r${r}.log" 2>&1
    echo "$(ts)   pseudo lines: $(wc -l < pseudo/${base}_r${r}.txt 2>/dev/null || echo 0)" | tee -a outputs/sweep_status.txt
    tag="${base}_on"; [ "$ROUNDS" -gt 1 ] && tag="${base}_r${r}"
    train_round "$M" "$tag" "${base}_m${r}.pt" "pseudo/${base}_r${r}.txt"
    prev="ckpt/${base}_m${r}.pt"
  done
done
echo "$(ts) SWEEP DONE" | tee -a outputs/sweep_status.txt; echo 0 > outputs/DONE
