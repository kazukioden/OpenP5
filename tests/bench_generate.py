"""
A/B harness for the KV-cache fix in P5_T5.prepare_inputs_for_generation.

Invariant under test: enabling the KV cache is a PERFORMANCE-only change, so
the constrained beam-search output (top-k item ids + scores) must be identical
before and after the fix.

Usage:
    python tests/bench_generate.py --device cpu  --out /tmp/gen_before.json
    # (apply fix)
    python tests/bench_generate.py --device cpu  --out /tmp/gen_after.json
    python tests/bench_generate.py --compare /tmp/gen_before.json /tmp/gen_after.json

Correctness is checked on CPU (fully deterministic, exact match expected).
Timing is meaningful on --device mps.
"""
import os, sys, json, time, argparse
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src", "src_t5"))

import torch
from transformers import AutoTokenizer, T5Config


def build_args(dataset, task):
    return SimpleNamespace(
        data_path=os.path.join(REPO, "data"),
        dataset=dataset,
        task=task,
        item_indexing="sequential",
        sequential_order="original",
        collaborative_token_size=200,
        collaborative_cluster=20,
        collaborative_last_token="sequential",
        collaborative_float32=0,
        prompt_file=os.path.join(REPO, "prompt.txt"),
        max_his=20,
        his_prefix=1,
        his_sep=" , ",
        test_prompt="seen:0",
        test_filtered=0,
    )


def run(device, ckpt, dataset, task, n_samples, num_beams, out_path):
    import utils.generation_trie as gt
    from data.TestDataset import TestDataset
    from processor.Collator import Collator
    from model.P5_T5 import P5_T5

    torch.manual_seed(2023)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    config = T5Config.from_pretrained("t5-small")
    model = P5_T5.from_pretrained("t5-small", config=config)
    model.resize_token_embeddings(len(tokenizer))
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    args = build_args(dataset, task)
    ds = TestDataset(args, dataset, task)

    # fixed slice of test inputs
    idxs = list(range(min(n_samples, len(ds))))
    batch = [ds[i] for i in idxs]
    collator = Collator(tokenizer)
    input_ids, attn, wwids, out_ids, out_attn = collator(batch)
    input_ids, attn, wwids = input_ids.to(device), attn.to(device), wwids.to(device)

    candidates = ds.all_items
    candidate_trie = gt.Trie(
        [[0] + tokenizer.encode(f"{dataset} item_{c}") for c in candidates]
    )
    prefix_allowed = gt.prefix_allowed_tokens_fn(candidate_trie)

    torch.manual_seed(2023)
    t0 = time.time()
    with torch.no_grad():
        pred = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            whole_word_ids=wwids,
            max_length=30,
            prefix_allowed_tokens_fn=prefix_allowed,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            output_scores=True,
            return_dict_in_generate=True,
        )
    elapsed = time.time() - t0

    gen = tokenizer.batch_decode(pred["sequences"], skip_special_tokens=True)
    scores = [round(float(s), 4) for s in pred["sequences_scores"]]
    # group top-k per input
    results = []
    for i in range(len(idxs)):
        seg = gen[i * num_beams:(i + 1) * num_beams]
        sc = scores[i * num_beams:(i + 1) * num_beams]
        results.append({"input": ds.data["input"][idxs[i]], "preds": seg, "scores": sc})

    payload = {
        "device": device, "task": task, "n": len(idxs), "num_beams": num_beams,
        "elapsed_sec": round(elapsed, 3), "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{task}] device={device} n={len(idxs)} beams={num_beams} "
          f"elapsed={elapsed:.2f}s -> {out_path}")
    return payload


def compare(before_path, after_path):
    b = json.load(open(before_path))
    a = json.load(open(after_path))
    assert b["n"] == a["n"], "sample count differs"
    mism = 0
    for i, (rb, ra) in enumerate(zip(b["results"], a["results"])):
        if rb["preds"] != ra["preds"]:
            mism += 1
            print(f"  [MISMATCH preds] sample {i}")
            print(f"    before: {rb['preds']}")
            print(f"    after : {ra['preds']}")
        elif rb["scores"] != ra["scores"]:
            # scores differ but predictions same -> flag (fp noise vs real)
            print(f"  [scores differ, preds same] sample {i}: "
                  f"{rb['scores']} vs {ra['scores']}")
    print("=" * 60)
    if mism == 0:
        print(f"PASS: predictions identical across all {b['n']} samples")
    else:
        print(f"FAIL: {mism}/{b['n']} samples changed predictions")
    print(f"time: before={b['elapsed_sec']}s  after={a['elapsed_sec']}s  "
          f"speedup={b['elapsed_sec']/max(a['elapsed_sec'],1e-9):.2f}x")
    return mism == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default=os.path.join(REPO, "checkpoint", "tiny.pt"))
    ap.add_argument("--dataset", default="ML100K_tiny")
    ap.add_argument("--task", default="sequential")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--beams", type=int, default=10)
    ap.add_argument("--out", default="/tmp/gen.json")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    a = ap.parse_args()
    if a.compare:
        ok = compare(a.compare[0], a.compare[1])
        sys.exit(0 if ok else 1)
    else:
        run(a.device, a.ckpt, a.dataset, a.task, a.n, a.beams, a.out)
