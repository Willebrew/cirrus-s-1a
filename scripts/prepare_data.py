#!/usr/bin/env python3
"""Stream, tokenize and shard a pretraining mix. Resumable: rerun the same
command after any interruption and it continues where it stopped.

    pip install tokenizers datasets zstandard
    python scripts/prepare_data.py --tokens 3e9 --out data/ablate     # an ablation-sized mix (~6 GB)
    python scripts/prepare_data.py --tokens 200e9 --out data/lean     # the full lean recipe (~400 GB)

Each (source, worker) pair streams its slice of the dataset from the Hub and
writes uint16 shards of ~100M tokens (200 MB) cut at document boundaries,
straight to disk. Rank 0 of each source also writes a held-out val file.

Memory: --workers is how many slices the data is split into (fixed for the
life of an output directory); --procs is how many run at once (free to
change between runs). Each process needs roughly PROC_GB of RAM, so by
default --procs is chosen to fit --max-mem-gb. The Hub download is usually
the bottleneck, so a few processes lose little.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cirrus.data.sources import MIXES, SOURCES  # noqa: E402

# Per-process RAM: ~0.2 GB for the interpreter, pyarrow and tokenizer, plus one decoded parquet row
# group of the rendered columns (Arrow decodes a whole column chunk even when reading small batches).
# ~1 GB measured on a 375 MB single-row-group file; 2 GB leaves headroom for SYNTH's larger files.
PROC_GB = 2.0


def worker(name: str, rank: int, world: int, budget: int, a: dict) -> dict:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # parallelism comes from processes
    from cirrus.data.shards import write_source
    from cirrus.data.tokenizer import CirrusTokenizer

    tok = CirrusTokenizer(a["tokenizer"])
    src = SOURCES[name]
    st = write_source(
        stream=lambda state: src.stream(rank, world, state),  # state: exact resume point, or None
        encode_batch=tok.encode_batch,
        eos_id=tok.eos_id,
        out=Path(a["out"]),
        name=name,
        rank=rank,
        budget=budget,
        val_tokens=a["val_tokens"],
        shard_tokens=a["shard_tokens"],
        max_passes=a["max_passes"].get(name, 1),
        log=lambda m: print(m, flush=True),
    )
    return {"source": name, "rank": rank, "tokens": st.tokens, "budget": budget, "exhausted": st.exhausted}


def reconcile_manifest(mpath: Path, manifest: dict) -> dict:
    """Merge a new run's manifest with the one already in --out (if any).

    The mix, budgets and tokenizer must match. The slice count is kept from
    the first run: each slice's resume state is a position within *its own*
    files, so re-slicing would orphan finished work and re-read or skip data.
    How many slices run at once (--procs) is free to change.
    """
    if not mpath.exists():
        return manifest
    old = json.loads(mpath.read_text())
    for k in ("mix", "budgets", "tokenizer_hash"):
        if old.get(k) != manifest[k]:
            sys.exit(f"{mpath.parent} was started with a different {k} ({old.get(k)} vs {manifest[k]}); "
                     "use a new --out or the original arguments")
    if old["workers"] != manifest["workers"]:
        print(f"note: {mpath.parent} was split into {sum(old['workers'].values())} slices; keeping that split "
              "(--workers only applies to a new --out; use --procs to run fewer at once)")
    return {**manifest, "workers": old["workers"], "written": old.get("written", {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="lean", choices=sorted(MIXES))
    ap.add_argument("--tokens", type=float, required=True, help="total train tokens across sources")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "tokenizer" / "cirrus-32k.json"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                    help="slices to split the data into (locked once an --out directory is started)")
    ap.add_argument("--procs", type=int, default=None,
                    help="worker processes running at once (default: as many as fit --max-mem-gb)")
    ap.add_argument("--max-mem-gb", type=float, default=20.0, help="RAM budget for all worker processes")
    ap.add_argument("--val-tokens", type=float, default=2e6, help="held-out tokens per source")
    ap.add_argument("--shard-tokens", type=float, default=1e8, help="tokens per shard file (resume granularity)")
    ap.add_argument("--sources", default=None,
                    help="comma-separated subset of the mix to write here (split the work across machines, "
                         "then merge the train/ and val/ directories)")
    args = ap.parse_args()

    mix = MIXES[args.mix]
    budgets = mix.budgets(args.tokens)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # workers per source in proportion to its budget, at least one each
    share = {k: max(1, round(args.workers * b / args.tokens)) for k, b in budgets.items()}
    tok_bytes = Path(args.tokenizer).read_bytes()
    manifest = {"mix": args.mix, "weights": mix.weights, "budgets": budgets, "workers": share,
                "tokenizer": str(args.tokenizer), "tokenizer_bytes": len(tok_bytes),
                "tokenizer_hash": hashlib.sha256(tok_bytes).hexdigest()[:16]}
    mpath = out / "manifest.json"
    manifest = reconcile_manifest(mpath, manifest)
    share = manifest["workers"]
    mpath.write_text(json.dumps(manifest, indent=1))

    a = {"tokenizer": args.tokenizer, "out": str(out), "val_tokens": int(args.val_tokens),
         "shard_tokens": int(args.shard_tokens), "max_passes": mix.max_passes}
    mine = args.sources.split(",") if args.sources else list(budgets)
    unknown = set(mine) - set(budgets)
    if unknown:
        sys.exit(f"--sources not in the {args.mix} mix: {sorted(unknown)}")
    tasks = [(name, r, share[name], budgets[name] // share[name] + (budgets[name] % share[name] if r == 0 else 0))
             for name in mine for r in range(share[name])]
    procs = args.procs or max(1, int(args.max_mem_gb // PROC_GB))
    procs = min(procs, len(tasks))
    print(f"{args.mix} mix, {args.tokens / 1e9:.1f}B tokens -> {out}, {len(tasks)} slices, {procs} at a time "
          f"(~{procs * PROC_GB:.0f} GB RAM)")
    for name in mine:
        print(f"   {name:18s} {budgets[name] / 1e9:7.2f}B tokens, {share[name]} slices")
    t0 = time.time()
    results = []
    # a fresh process per slice, so memory a finished slice leaves behind (arrow pools) is returned
    with ProcessPoolExecutor(max_workers=procs, max_tasks_per_child=1) as ex:
        futs = [ex.submit(worker, *t, a) for t in tasks]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"done {r['source']}/{r['rank']}: {r['tokens']:,} tokens"
                  f"{' (data exhausted)' if r['exhausted'] else ''} [{(time.time() - t0) / 3600:.2f} h]", flush=True)
    total = sum(r["tokens"] for r in results)
    written = {k: sum(r["tokens"] for r in results if r["source"] == k) for k in mine}
    manifest["written"] = {**manifest.get("written", {}), **written}
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"\n{total / 1e9:.2f}B tokens written in {(time.time() - t0) / 3600:.2f} h")
    short = {k: v for k, v in written.items() if v < 0.98 * budgets[k]}
    if short:
        print("sources below budget (the mix is skewed toward the others):", short)


if __name__ == "__main__":
    main()
