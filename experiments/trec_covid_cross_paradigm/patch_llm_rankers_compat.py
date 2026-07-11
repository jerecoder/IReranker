#!/usr/bin/env python3
"""Patch the pinned 2023 llm-rankers tokenizer API for modern Transformers."""

from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_llm_rankers_compat.py external/llm-rankers/run.py")
    root = Path(sys.argv[1]).resolve().parent
    for relative in ("llmrankers/setwise.py", "llmrankers/listwise.py"):
        path = root / relative
        text = path.read_text(encoding="utf-8")
        patched = text.replace("self.tokenizer.batch_encode_plus(", "self.tokenizer(")
        path.write_text(patched, encoding="utf-8")
        print(f"Compatibility checked: {path}")


if __name__ == "__main__":
    main()
