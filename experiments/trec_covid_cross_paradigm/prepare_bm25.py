#!/usr/bin/env python3
"""Create the common TREC-COVID BM25 top-100 input run."""

from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/external/beir/bm25-runs/run.beir.bm25-flat.trec-covid.txt"
OUTPUT = ROOT / "experiments/trec_covid_cross_paradigm/runs/bm25.trec-covid.txt"


def main() -> None:
    counts: dict[str, int] = defaultdict(int)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(encoding="utf-8") as source, OUTPUT.open("w", encoding="utf-8") as output:
        for line in source:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid = parts[0]
            if counts[qid] < 100:
                output.write(line)
                counts[qid] += 1
    if len(counts) != 50 or any(count != 100 for count in counts.values()):
        raise RuntimeError(f"Expected 50 queries x 100 documents; got {dict(counts)}")
    print(f"Wrote {sum(counts.values())} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
