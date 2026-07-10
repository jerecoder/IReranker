from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET

QRELS_HEADER = ["query-id", "corpus-id", "score"]


def is_public_task_dataset(dataset_name: str) -> bool:
    return dataset_name in {"dl-2019", "dl-2020", "robust04", "trec-news"}


def build_public_task_dataset(dataset_name: str, out_root: Path) -> Path:
    if dataset_name == "dl-2019":
        return build_dl(
            "dl-2019",
            queries_url=(
                "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
                "msmarco-test2019-queries.tsv.gz"
            ),
            qrels_url="https://trec.nist.gov/data/deep/2019qrels-pass.txt",
            out_root=out_root,
        )
    if dataset_name == "dl-2020":
        return build_dl(
            "dl-2020",
            queries_url=(
                "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
                "msmarco-test2020-queries.tsv.gz"
            ),
            qrels_url="https://trec.nist.gov/data/deep/2020qrels-pass.txt",
            out_root=out_root,
        )
    if dataset_name == "robust04":
        return build_robust04(out_root)
    if dataset_name == "trec-news":
        return build_trec_news_2019(out_root)
    raise ValueError(f"Unsupported public task dataset: {dataset_name}")


def has_public_task_layout(base_dir: Path) -> bool:
    return (base_dir / "queries.jsonl").exists() and (base_dir / "qrels" / "test.tsv").exists()


def http_get(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Encoding": "gzip",
        },
        method="GET",
    )
    with urllib.request.urlopen(req) as r:
        data = r.read()
        enc = (r.headers.get("Content-Encoding") or "").lower()

    if "gzip" in enc:
        data = gzip.decompress(data)
    return data


def ensure_beir_layout(base_dir: Path) -> None:
    qrels_dir = base_dir / "qrels"
    qrels_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        path = qrels_dir / f"{split}.tsv"
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(QRELS_HEADER)


def write_qrels_trec_to_tsv(out_path: Path, trec_qrels_text: str) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(QRELS_HEADER)
        for line in trec_qrels_text.splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, docid, rel = parts[0], parts[2], parts[3]
            writer.writerow([qid, docid, rel])


def write_queries_tsv_gz_to_jsonl(out_path: Path, payload: bytes) -> None:
    if len(payload) >= 2 and payload[0] == 0x1F and payload[1] == 0x8B:
        text = gzip.decompress(payload).decode("utf-8", errors="replace")
    else:
        text = payload.decode("utf-8", errors="replace")

    with out_path.open("w", encoding="utf-8") as f:
        for line in text.splitlines():
            if not line.strip():
                continue
            qid, qtext = line.split("\t", 1)
            obj = {"_id": qid, "text": qtext, "metadata": {}}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_queries_jsonl(out_path: Path, rows: list[dict[str, object]]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_dl(dataset_name: str, queries_url: str, qrels_url: str, out_root: Path) -> Path:
    base_dir = out_root / dataset_name
    ensure_beir_layout(base_dir)

    queries_gz = http_get(queries_url)
    write_queries_tsv_gz_to_jsonl(base_dir / "queries.jsonl", queries_gz)

    qrels_txt = http_get(qrels_url).decode("utf-8", errors="replace")
    write_qrels_trec_to_tsv(base_dir / "qrels" / "test.tsv", qrels_txt)
    return base_dir


def _parse_top_topics(topics_raw: str) -> list[dict[str, object]]:
    queries: list[dict[str, object]] = []
    blocks = topics_raw.split("<top>")
    for block in blocks[1:]:
        num = ""
        docid = ""
        url = ""
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("<num>"):
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
    return queries


def _parse_topic_xml(topics_raw: str) -> list[dict[str, object]]:
    blocks = re.findall(
        r"<topic\b[^>]*>.*?</topic>",
        topics_raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    queries: list[dict[str, object]] = []
    for block in blocks:
        topic = ET.fromstring(block)
        qid = (topic.get("number") or topic.findtext("number") or "").strip()
        docid = (
            topic.findtext("docid")
            or topic.findtext("doc_id")
            or topic.findtext("docno")
            or ""
        ).strip()
        url = (topic.findtext("url") or "").strip()
        if qid:
            queries.append({"_id": qid, "doc_id": docid, "url": url, "metadata": {}})
    return queries


def build_trec_news_2019(out_root: Path) -> Path:
    base_dir = out_root / "trec-news"
    ensure_beir_layout(base_dir)

    topics_url = "https://trec.nist.gov/data/news/2019/newsir19-background-linking-topics.xml"
    qrels_url = "https://trec.nist.gov/data/news/2019/newsir19-qrels-background.txt"

    topics_raw = http_get(topics_url).decode("utf-8", errors="replace")
    qrels_txt = http_get(qrels_url).decode("utf-8", errors="replace")

    low = topics_raw.lower()
    if "<top>" in low:
        queries = _parse_top_topics(topics_raw)
    elif "<topic" in low:
        queries = _parse_topic_xml(topics_raw)
    else:
        (base_dir / "topics_download_debug.txt").write_text(topics_raw[:5000], encoding="utf-8")
        raise RuntimeError(
            "TREC News topics download did not contain <top> or <topic>. "
            "Saved first 5000 chars to trec-news/topics_download_debug.txt"
        )

    write_queries_jsonl(base_dir / "queries.jsonl", queries)
    write_qrels_trec_to_tsv(base_dir / "qrels" / "test.tsv", qrels_txt)
    return base_dir


def parse_robust_topics_from_testset_gz(raw_gz: bytes) -> list[dict[str, object]]:
    text = gzip.decompress(raw_gz).decode("utf-8", errors="replace")
    blocks = text.split("<top>")
    queries: list[dict[str, object]] = []
    for block in blocks[1:]:
        num = ""
        title = ""
        for line in block.splitlines():
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


def build_robust04(out_root: Path) -> Path:
    base_dir = out_root / "robust04"
    ensure_beir_layout(base_dir)

    topics_gz_url = "https://trec.nist.gov/data/robust/04.testset.gz"
    qrels_url = "https://trec.nist.gov/data/robust/qrels.robust2004.txt"

    raw_topics_gz = http_get(topics_gz_url)
    qrels_txt = http_get(qrels_url).decode("utf-8", errors="replace")

    queries = parse_robust_topics_from_testset_gz(raw_topics_gz)
    write_queries_jsonl(base_dir / "queries.jsonl", queries)
    write_qrels_trec_to_tsv(base_dir / "qrels" / "test.tsv", qrels_txt)
    return base_dir
