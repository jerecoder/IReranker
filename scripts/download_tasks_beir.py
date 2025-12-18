import csv
import gzip
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

QRELS_HEADER = ["query-id", "corpus-id", "score"]

def http_get(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Encoding": "gzip",   # allow gzip transfer encoding
        },
        method="GET",
    )
    with urllib.request.urlopen(req) as r:
        data = r.read()
        enc = (r.headers.get("Content-Encoding") or "").lower()

    # Only decompress if the *transfer encoding* is gzip
    if "gzip" in enc:
        data = gzip.decompress(data)

    return data

def ensure_beir_layout(base_dir: Path):
    qrels_dir = base_dir / "qrels"
    qrels_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        p = qrels_dir / f"{split}.tsv"
        if not p.exists():
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(QRELS_HEADER)

def write_qrels_trec_to_tsv(out_path: Path, trec_qrels_text: str):
    # TREC qrels: qid iter docid rel
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(QRELS_HEADER)
        for line in trec_qrels_text.splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, docid, rel = parts[0], parts[2], parts[3]
            w.writerow([qid, docid, rel])

def write_queries_tsv_gz_to_jsonl(out_path: Path, payload: bytes):
    # If payload is gzip content, it starts with gzip magic bytes 1f 8b
    if len(payload) >= 2 and payload[0] == 0x1F and payload[1] == 0x8B:
        text = gzip.decompress(payload).decode("utf-8", errors="replace")
    else:
        # Already plain TSV (e.g., because http_get decompressed transfer-encoding)
        text = payload.decode("utf-8", errors="replace")

    with out_path.open("w", encoding="utf-8") as f:
        for line in text.splitlines():
            if not line.strip():
                continue
            qid, qtext = line.split("\t", 1)
            obj = {"_id": qid, "text": qtext, "metadata": {}}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_queries_jsonl(out_path: Path, rows):
    with out_path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def build_dl(dataset_name: str, queries_url: str, qrels_url: str, out_root: Path):
    base_dir = out_root / dataset_name
    ensure_beir_layout(base_dir)

    # queries.jsonl
    queries_gz = http_get(queries_url)
    write_queries_tsv_gz_to_jsonl(base_dir / "queries.jsonl", queries_gz)

    # qrels/test.tsv (train/dev stay header-only)
    qrels_txt = http_get(qrels_url).decode("utf-8", errors="replace")
    write_qrels_trec_to_tsv(base_dir / "qrels" / "test.tsv", qrels_txt)

    print(f"[{dataset_name}] wrote queries.jsonl + qrels/{{train,dev,test}}.tsv -> {base_dir}")

import re
import xml.etree.ElementTree as ET

def _extract_topic_blocks(text: str):
    return re.findall(r"<topic\b[^>]*>.*?</topic>", text, flags=re.DOTALL | re.IGNORECASE)

def _parse_topic_block(block: str):
    # Parse a single topic block as XML
    t = ET.fromstring(block)
    qid = (t.get("number") or t.findtext("number") or "").strip()
    docid = (t.findtext("docid") or t.findtext("doc_id") or t.findtext("docno") or "").strip()
    url = (t.findtext("url") or "").strip()
    return qid, docid, url

def build_trec_news_2019(out_root: Path):
    base_dir = out_root / "trec-news"
    ensure_beir_layout(base_dir)

    topics_url = "https://trec.nist.gov/data/news/2019/newsir19-background-linking-topics.xml"
    qrels_url  = "https://trec.nist.gov/data/news/2019/newsir19-qrels-background.txt"

    topics_raw = http_get(topics_url).decode("utf-8", errors="replace")
    qrels_txt  = http_get(qrels_url).decode("utf-8", errors="replace")

    low = topics_raw.lower()

    # --- Handle TREC <top> format (what you're actually getting) ---
    if "<top>" in low:
        blocks = topics_raw.split("<top>")
        queries = []
        for b in blocks[1:]:
            num = ""
            docid = ""
            url = ""
            for line in b.splitlines():
                line = line.strip()

                if line.startswith("<num>"):
                    # e.g. "<num> Number: 826 </num>"
                    num = (
                        line.replace("<num>", "")
                            .replace("</num>", "")
                            .replace("Number:", "")
                            .strip()
                    )
                elif line.startswith("<docid>"):
                    docid = line.replace("<docid>", "").replace("</docid>", "").strip()
                elif line.startswith("<url>"):
                    url = line.replace("<url>", "").replace("</url>", "").strip()

            if num:
                queries.append({"_id": num, "doc_id": docid, "url": url, "metadata": {}})

        write_queries_jsonl(base_dir / "queries.jsonl", queries)
        write_qrels_trec_to_tsv(base_dir / "qrels" / "test.tsv", qrels_txt)
        print(f"[trec-news] wrote queries.jsonl (ids only) + qrels -> {base_dir}")
        return

    # --- Fallback: if it ever really is <topic> XML ---
    if "<topic" in low:
        blocks = re.findall(r"<topic\b[^>]*>.*?</topic>", topics_raw, flags=re.DOTALL | re.IGNORECASE)
        queries = []
        for block in blocks:
            t = ET.fromstring(block)
            qid   = (t.get("number") or t.findtext("number") or "").strip()
            docid = (t.findtext("docid") or t.findtext("doc_id") or t.findtext("docno") or "").strip()
            url   = (t.findtext("url") or "").strip()
            if qid:
                queries.append({"_id": qid, "doc_id": docid, "url": url, "metadata": {}})
        write_queries_jsonl(base_dir / "queries.jsonl", queries)
        write_qrels_trec_to_tsv(base_dir / "qrels" / "test.tsv", qrels_txt)
        print(f"[trec-news] wrote queries.jsonl (ids only) + qrels -> {base_dir}")
        return

    # Debug dump if neither pattern matches
    (base_dir / "topics_download_debug.txt").write_text(topics_raw[:5000], encoding="utf-8")
    raise RuntimeError(
        "TREC News topics download did not contain <top> or <topic>. "
        "Saved first 5000 chars to trec-news/topics_download_debug.txt"
    )

def parse_robust_topics_from_testset_gz(raw_gz: bytes):
    text = gzip.decompress(raw_gz).decode("utf-8", errors="replace")
    blocks = text.split("<top>")
    queries = []
    for b in blocks[1:]:
        num = ""
        title = ""
        for line in b.splitlines():
            line = line.strip()
            if line.startswith("<num>"):
                num = line.replace("<num>", "").replace("Number:", "").strip()
            elif line.startswith("<title>"):
                title = line.replace("<title>", "").strip()
                if num:
                    break
        if num:
            queries.append({"_id": num, "text": title, "metadata": {}})
    return queries

def build_robust04(out_root: Path):
    base_dir = out_root / "robust04"
    ensure_beir_layout(base_dir)

    topics_gz_url = "https://trec.nist.gov/data/robust/04.testset.gz"
    qrels_url     = "https://trec.nist.gov/data/robust/qrels.robust2004.txt"

    raw_topics_gz = http_get(topics_gz_url)
    qrels_txt     = http_get(qrels_url).decode("utf-8", errors="replace")

    queries = parse_robust_topics_from_testset_gz(raw_topics_gz)
    write_queries_jsonl(base_dir / "queries.jsonl", queries)
    write_qrels_trec_to_tsv(base_dir / "qrels" / "test.tsv", qrels_txt)

    print(f"[robust04] wrote queries.jsonl + qrels -> {base_dir}")

def main():
    out_root = Path("data/external/beir")

    # DL passage tasks (queries + qrels only; corpus is MS MARCO passage collection)
    build_dl(
        "dl-2019",
        queries_url="https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
        qrels_url="https://trec.nist.gov/data/deep/2019qrels-pass.txt",
        out_root=out_root,
    )
    build_dl(
        "dl-2020",
        queries_url="https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
        qrels_url="https://trec.nist.gov/data/deep/2020qrels-pass.txt",
        out_root=out_root,
    )

    # Licensed corpora: tasks are public, corpora are not.
    build_trec_news_2019(out_root)
    build_robust04(out_root)

if __name__ == "__main__":
    main()
