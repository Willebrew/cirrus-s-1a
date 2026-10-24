#!/usr/bin/env python3
"""Measure tau (accepted tokens per verify step) on search-style answers.

cirrus-s-1a isn't trained yet, so this uses existing open models as proxies:
each one answers real questions from real retrieved sources in our format
(numbered sources, organized answer, [n] citations), with greedy decoding.
Because greedy draft-and-verify emits exactly the greedy tokens, replaying
our prompt-lookup drafter over each answer gives the *exact* tau of copy
drafting for that model (cirrus/decoding/tau.py). The MTP head's share is
modeled on a grid of acceptance rates.

Every tau is converted to decode tok/s with the weight stream measured on
the Spark in run 6 (cirrus/perf/estimate.py).

    pip install transformers datasets accelerate
    python scripts/measure_tau.py                       # defaults: 2 models x 2 styles, HotpotQA, 100 questions
    python scripts/measure_tau.py --source msmarco --n 200
    python scripts/measure_tau.py --models Qwen/Qwen3-8B --styles extractive

Read the results with two caveats: the proxy's tokenizer is not cirrus's
(32K BPE, not trained yet), and cirrus will be *trained* toward the
extractive style, so "extractive" is the closer proxy for its behaviour.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cirrus.decoding.tau import TauStats, copy_fraction, simulate  # noqa: E402
from cirrus.perf.estimate import RUN6_STREAM_US, tok_s_from_stream  # noqa: E402

STYLES = {
    "default": (
        "You are a search engine. Answer the user's query using ONLY the numbered sources. "
        "Write an organized answer (short headings or bullets where useful). End every sentence "
        "that states a fact with the source number(s) in brackets, like [2] or [1, 3]. If the "
        "sources do not answer the query, say so."
    ),
    "extractive": (
        "You are a search engine. Answer the user's query using ONLY the numbered sources. "
        "Write an organized answer (short headings or bullets where useful). Reuse the sources' "
        "exact wording for names, numbers, dates, titles and key phrases instead of paraphrasing. "
        "End every sentence that states a fact with the source number(s) in brackets, like [2] or "
        "[1, 3]. If the sources do not answer the query, say so."
    ),
}

# Drafting configurations: (label, max lookup draft, MTP acceptance chain, mode)
MTP_GRID = {"none": (), "mtp70/50": (0.7, 0.5), "mtp80/60": (0.8, 0.6), "mtp90/75": (0.9, 0.75)}
DRAFT_LENGTHS = (3, 7, 12)  # verify widths 4, 8, 13 (the widths the kernel was measured at)


# ----------------------------------------------------------------- data
def load_examples(source: str, n: int, max_chars: int, file: str | None):
    if source == "file":
        rows = [json.loads(l) for l in Path(file).read_text().splitlines() if l.strip()]
        return [{"query": r["query"], "sources": r["sources"][:10]} for r in rows[:n]]
    from datasets import load_dataset

    out = []
    if source == "hotpotqa":
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        for ex in ds.select(range(min(n, len(ds)))):
            srcs = [{"title": t, "url": "", "text": " ".join(s)[:max_chars]}
                    for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])]
            out.append({"query": ex["question"], "sources": srcs})
    elif source == "msmarco":
        ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
        for ex in ds:
            p = ex["passages"]
            if not p["passage_text"] or ex.get("answers") == ["No Answer Present."]:
                continue
            srcs = [{"title": "", "url": u, "text": t[:max_chars]} for t, u in zip(p["passage_text"], p["url"])]
            out.append({"query": ex["query"], "sources": srcs[:10]})
            if len(out) >= n:
                break
    else:
        raise ValueError(source)
    return out


def render_user(ex) -> str:
    lines = [f"Query: {ex['query']}", "", "Sources:"]
    for i, s in enumerate(ex["sources"], 1):
        head = " | ".join(x for x in (s.get("title", ""), s.get("url", "")) if x)
        lines.append(f"[{i}] {head}\n{s['text']}")
    return "\n".join(lines)


# ----------------------------------------------------------- generation
def generate(model_id: str, prompts: list[list[dict]], max_new: int, batch: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for m in prompts]
    results = []
    t0 = time.time()
    for b in range(0, len(texts), batch):
        enc = tok(texts[b : b + batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, pad_token_id=tok.pad_token_id)
        width = enc["input_ids"].shape[1]
        for r in range(gen.shape[0]):
            mask = enc["attention_mask"][r].bool()
            prompt_ids = enc["input_ids"][r][mask].tolist()
            out = gen[r, width:].tolist()
            # cut at the first end-of-sequence token (keep it: the model does emit it)
            stops = {tok.eos_token_id, tok.pad_token_id} | set(getattr(model.generation_config, "eos_token_id", []) or [])
            for j, t in enumerate(out):
                if t in stops:
                    out = out[: j + 1]
                    break
            results.append((prompt_ids, out, tok.decode(out, skip_special_tokens=True)))
        done = min(b + batch, len(texts))
        print(f"   {model_id}: {done}/{len(texts)} answers ({time.time() - t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return results


# ------------------------------------------------------------- analysis
def analyze(samples) -> dict:
    rows = {}
    for k in DRAFT_LENGTHS:
        for mlabel, chain in MTP_GRID.items():
            for mode in (("fallback", "tree") if chain else ("fallback",)):
                agg, per = TauStats(), []
                for idx, (p, o, _) in enumerate(samples):
                    st = simulate(p, o, max_draft=k, mtp=chain, mode=mode, seed=idx)
                    agg.merge(st)
                    if st.steps:
                        per.append(st.tau)
                verify = k + 1 + (2 if mode == "tree" else 0)
                tau = agg.tau
                q = statistics.quantiles(per, n=10) if len(per) >= 10 else [tau] * 9
                rows[f"k{k}_{mlabel}_{mode}"] = {
                    "draft_len": k, "mtp": mlabel, "mode": mode, "verify_width": verify,
                    "tau": round(tau, 3), "tau_p10": round(q[0], 2), "tau_p50": round(q[4], 2), "tau_p90": round(q[8], 2),
                    "lookup_step_share": round(agg.lookup_steps / max(1, agg.steps), 3),
                    "tok_s_32k": round(tok_s_from_stream(RUN6_STREAM_US, tau, verify, 32768)),
                    "tok_s_4k": round(tok_s_from_stream(RUN6_STREAM_US, tau, verify, 4096)),
                }
    lengths = [len(o) for _, o, _ in samples]
    texts = [t for _, _, t in samples]
    return {
        "n": len(samples),
        "answer_tokens_mean": round(statistics.mean(lengths), 1),
        "prompt_tokens_mean": round(statistics.mean(len(p) for p, _, _ in samples), 1),
        "copy_fraction_4gram": round(statistics.mean(copy_fraction(p, o) for p, o, _ in samples), 3),
        "citations_per_answer": round(statistics.mean(len(re.findall(r"\[\d+(?:,\s*\d+)*\]", t)) for t in texts), 2),
        "configs": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"])
    ap.add_argument("--styles", nargs="+", default=["default", "extractive"], choices=sorted(STYLES))
    ap.add_argument("--source", default="hotpotqa", choices=["hotpotqa", "msmarco", "file"])
    ap.add_argument("--file", help="JSONL of {query, sources:[{title,url,text}]} for --source file")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-chars", type=int, default=1200, help="per-source text cap")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "results" / "local"))
    args = ap.parse_args()

    examples = load_examples(args.source, args.n, args.max_chars, args.file)
    print(f"loaded {len(examples)} examples from {args.source}", flush=True)
    report = {"source": args.source, "n": len(examples), "host": socket.gethostname(),
              "stream_us_used": RUN6_STREAM_US, "results": {}, "samples": {}}
    for model_id in args.models:
        for style in args.styles:
            prompts = [[{"role": "system", "content": STYLES[style]}, {"role": "user", "content": render_user(ex)}]
                       for ex in examples]
            samples = generate(model_id, prompts, args.max_new, args.batch)
            res = analyze(samples)
            key = f"{model_id}|{style}"
            report["results"][key] = res
            report["samples"][key] = [{"query": ex["query"], "answer": t} for ex, (_, _, t) in zip(examples[:3], samples[:3])]
            c = res["configs"]
            print(f"\n== {model_id} ({style}): {res['n']} answers, {res['answer_tokens_mean']} tokens avg, "
                  f"4-gram copy {res['copy_fraction_4gram']:.0%}, {res['citations_per_answer']} citations/answer")
            print(f"   {'config':28s} {'tau':>6s} {'p10':>5s} {'p50':>5s} {'p90':>5s} {'verify':>6s} {'tok/s@32K':>10s}")
            for name, r in c.items():
                print(f"   {name:28s} {r['tau']:6.2f} {r['tau_p10']:5.2f} {r['tau_p50']:5.2f} {r['tau_p90']:5.2f} "
                      f"{r['verify_width']:6d} {r['tok_s_32k']:10,d}")
            best = max(c.values(), key=lambda r: r["tok_s_32k"])
            print(f"   best: {best['mtp']} {best['mode']} draft {best['draft_len']}: tau {best['tau']}, "
                  f"{best['tok_s_32k']:,} tok/s @32K")
            lookup_best = max((r for r in c.values() if r["mtp"] == "none"), key=lambda r: r["tok_s_32k"])
            print(f"   lookup only (no MTP head): tau {lookup_best['tau']}, {lookup_best['tok_s_32k']:,} tok/s @32K")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tau-{socket.gethostname()}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nresults written to {path}")


if __name__ == "__main__":
    main()
