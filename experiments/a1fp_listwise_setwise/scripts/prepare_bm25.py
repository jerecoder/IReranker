#!/usr/bin/env python3
"""Prepare top-100 BM25 runs for the A1fp listwise/setwise experiment."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "a1fp_listwise_setwise"

SOURCES = {
    "dl19": ROOT / "data" / "external" / "beir" / "bm25-runs" / "run.beir.bm25-flat.dl-2019.txt",
    "dl20": ROOT / "data" / "external" / "beir" / "bm25-runs" / "run.beir.bm25-flat.dl-2020.txt",
}

OUTPUTS = {
    "dl19": EXP_DIR / "runs" / "bm25.dl19.txt",
    "dl20": EXP_DIR / "runs" / "bm25.dl20.txt",
}


def write_top_k(source: Path, output: Path, k: int = 100) -> tuple[int, int]:
    if not source.exists():
        raise FileNotFoundError(f"Missing source BM25 run: {source}")

    counts: dict[str, int] = defaultdict(int)
    written = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="utf-8") as inp, output.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Malformed TREC line in {source}: {line!r}")
            qid = parts[0]
            if counts[qid] >= k:
                continue
            counts[qid] += 1
            written += 1
            out.write(line)

    underfilled = {qid: count for qid, count in counts.items() if count != k}
    if underfilled:
        raise ValueError(f"{output} has queries with != {k} docs: {underfilled}")
    return len(counts), written


def main() -> int:
    for dataset, source in SOURCES.items():
        queries, lines = write_top_k(source, OUTPUTS[dataset])
        print(f"{dataset}: wrote {lines} lines for {queries} queries to {OUTPUTS[dataset]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
