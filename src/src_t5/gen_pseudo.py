"""
Self-bootstrap — Grow + Filter (piece 2).

Load a FROZEN trained checkpoint and, for each TRAIN user, generate top-k next-
item candidates via constrained beam search over the item Trie. Keep a candidate
as a pseudo-label if it is:
  - a NEW item (not already in the user's full sequence) -> adds signal AND
    excludes the leave-one-out valid/test items => no eval leakage, and
  - confident enough (sequence score >= --gen_conf).
Write kept `(user_id <TAB> reindexed_item_id)` to --pseudo_out.
"""
import os, argparse, torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5Config

from data.MultiTaskDataset import MultiTaskDataset
from model.P5_T5 import P5_T5
from processor.Collator import calculate_whole_word_ids
import utils.generation_trie as gt
from utils import utils, initialization


def main():
    parser = argparse.ArgumentParser(description='self-bootstrap gen_pseudo')
    parser = utils.parse_global_args(parser)
    parser = MultiTaskDataset.parse_dataset_args(parser)
    parser.add_argument('--backbone', type=str, default='t5-small')
    parser.add_argument('--load_path', type=str, required=True, help='checkpoint to generate from (frozen)')
    parser.add_argument('--pseudo_out', type=str, required=True)
    parser.add_argument('--gen_topk', type=int, default=5, help='beam size / candidates per user')
    parser.add_argument('--pseudo_per_user', type=int, default=1, help='max kept pseudo per user')
    parser.add_argument('--gen_conf', type=float, default=-1e9, help='min sequence score to keep (lenient by default)')
    parser.add_argument('--gen_batch', type=int, default=32)
    parser.add_argument('--random_initialize', type=int, default=1)
    args, _ = parser.parse_known_args()
    args.rank = 0
    args.distributed = 0  # single-device indexing (avoid dist.barrier)
    args.pseudo_file = ''  # never load pseudo when building the source dataset

    device = (torch.device('cuda', 0) if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    ds = MultiTaskDataset(args, args.datasets, 'train')  # reindex, all_items, prompt, his_sep, max_his

    config = T5Config.from_pretrained(args.backbone)
    model = P5_T5.from_pretrained(args.backbone, config=config)
    if args.item_indexing == 'collaborative':
        tokenizer.add_tokens(ds.new_token)
    model.resize_token_embeddings(len(tokenizer))
    if args.random_initialize == 1:
        initialization.random_initialization(model, tokenizer, args.backbone)
    state = torch.load(args.load_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    all_items = ds.all_items
    all_items_set = set(all_items)
    candidate_trie = gt.Trie(
        [[0] + tokenizer.encode(f"{args.datasets} item_{c}") for c in all_items])
    prefix_allowed = gt.prefix_allowed_tokens_fn(candidate_trie)

    tmpl = ds.prompt['sequential']['seen']['0']
    maxh = ds.max_his if 'history' in ds.info else 0

    users, inputs, known = [], [], []
    for user in ds.reindex_user_seq_dict:
        full = ds.reindex_user_seq_dict[user]
        hist = full[:-2]  # train region (leave valid[-2]/test[-1] out)
        if len(hist) == 0:
            continue
        h = hist[-maxh:] if (maxh and maxh > 0) else hist
        one = {'dataset': args.datasets, 'user_id': user, 'target': ''}
        if 'history' in ds.info:
            one['history'] = ds.his_sep.join(
                ['item_' + x for x in h] if ds.prefix > 0 else h)
        users.append(user)
        inputs.append(tmpl['Input'].format(**one))
        known.append(set(full))  # exclude every item already in the sequence

    def parse_item(seq):
        if 'item_' not in seq:
            return None
        cand = seq.split('item_')[-1].replace(' ', '')
        return cand if cand in all_items_set else None

    kept_total = 0
    with open(args.pseudo_out, 'w') as out:
        for i in tqdm(range(0, len(inputs), args.gen_batch)):
            chunk = inputs[i:i + args.gen_batch]
            enc = tokenizer.batch_encode_plus(chunk, padding='longest', truncation=True, max_length=512)
            input_ids = torch.tensor(enc['input_ids']).to(device)
            attn = torch.tensor(enc['attention_mask']).to(device)
            wwids = torch.tensor(
                [calculate_whole_word_ids(tokenizer.convert_ids_to_tokens(x), x) for x in enc['input_ids']]
            ).to(device)
            with torch.no_grad():
                pred = model.generate(
                    input_ids=input_ids, attention_mask=attn, whole_word_ids=wwids,
                    max_length=30, prefix_allowed_tokens_fn=prefix_allowed,
                    num_beams=args.gen_topk, num_return_sequences=args.gen_topk,
                    output_scores=True, return_dict_in_generate=True)
            seqs = tokenizer.batch_decode(pred['sequences'], skip_special_tokens=True)
            scores = pred['sequences_scores'].tolist()
            for u in range(len(chunk)):
                user = users[i + u]
                seen = known[i + u]
                cands = seqs[u * args.gen_topk:(u + 1) * args.gen_topk]
                scs = scores[u * args.gen_topk:(u + 1) * args.gen_topk]
                kept = 0
                for c, s in zip(cands, scs):
                    if s < args.gen_conf:
                        continue
                    item = parse_item(c)
                    if item is None or item in seen:
                        continue
                    out.write(f"{user}\t{item}\n")
                    kept += 1
                    kept_total += 1
                    if kept >= args.pseudo_per_user:
                        break
    print(f"[gen_pseudo] wrote {kept_total} pseudo labels -> {args.pseudo_out}")


if __name__ == "__main__":
    main()
