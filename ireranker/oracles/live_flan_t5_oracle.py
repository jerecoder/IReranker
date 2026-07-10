from __future__ import annotations

import atexit
import json
from pathlib import Path
import pickle
import random
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from ireranker.config import EXTERNAL_DATA_DIR, logger
from ireranker.oracles.oracle import MatrixKey, Oracle


def _slug_model(model_name: str) -> str:
    slug = model_name.lower().strip().replace("/", "-")
    slug = re.sub(r"[^a-z0-9._-]+", "-", slug)
    return re.sub(r"-+", "-", slug).strip("-")


def _record_text(row: Mapping[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return text or title


class LiveFlanT5Oracle(Oracle):
    """Base oracle that computes FLAN-T5 pairwise comparisons on demand."""

    _model_cache: Dict[tuple[str, str], tuple[Any, Any, str, int, int]] = {}
    _cache_store: Dict[Path, Dict[MatrixKey, Mapping[str, Any]]] = {}
    _cache_dirty: Dict[Path, int] = {}
    _text_store: Dict[str, tuple[Dict[str, str], Dict[str, str]]] = {}
    _atexit_registered = False

    def __init__(
        self,
        *,
        model_name: str = "google/flan-t5-large",
        cache_path: Path | None = None,
        device: str | None = None,
        max_input_tokens: int = 512,
        cache_flush_interval: int = 20,
        cache_comparisons: bool,
        comparison_limit: int | None = None,
        comparison_limit_per_task: bool = False,
    ) -> None:
        super().__init__(
            comparison_limit=comparison_limit,
            comparison_limit_per_task=comparison_limit_per_task,
        )
        self.model_name = model_name
        self.model_label = _slug_model(model_name)
        self.cache_path = cache_path
        self.device = device
        self.max_input_tokens = int(max_input_tokens)
        self.cache_flush_interval = max(int(cache_flush_interval), 1)
        self._dataset: str | None = None
        self._split: str | None = None
        self._allowed_qids: set[str] | None = None
        self._queries: Dict[str, str] = {}
        self._corpus: Dict[str, str] = {}
        self._comparison_cache_file: Dict[MatrixKey, Mapping[str, Any]] | None = None
        self.enable_cache(cache_comparisons)

        if not LiveFlanT5Oracle._atexit_registered:
            atexit.register(LiveFlanT5Oracle.flush_all_caches)
            LiveFlanT5Oracle._atexit_registered = True

    def load_dataset(
        self,
        dataset: str,
        *,
        split: str = "test",
        query_ids: Optional[Iterable[str]] = None,
        matrix_model: Optional[str] = None,
    ) -> None:
        self._clear_cache()
        self.current_task = None
        self._dataset = dataset.lower().strip()
        self._split = split.lower().strip() if split else "test"
        self._allowed_qids = {str(qid) for qid in query_ids} if query_ids is not None else None
        self.cache_path = self.cache_path or self._default_cache_path(self._dataset, self._split)
        self._comparison_cache_file = self._load_cache(self.cache_path)
        self._load_query_and_corpus_text(self._dataset)
        logger.info(
            f"Using live FLAN-T5 oracle: model={self.model_name}, "
            f"cache={self.cache_path}"
        )

    def sample_lt(self, i: int, j: int) -> bool:
        raise NotImplementedError

    @classmethod
    def flush_all_caches(cls) -> None:
        for path in list(cls._cache_store.keys()):
            cls._save_cache(path)

    @classmethod
    def _load_cache(cls, path: Path) -> Dict[MatrixKey, Mapping[str, Any]]:
        path = path.resolve()
        if path in cls._cache_store:
            return cls._cache_store[path]

        if path.exists():
            with path.open("rb") as f:
                obj = pickle.load(f)
            if not isinstance(obj, dict):
                raise ValueError(f"Unexpected live comparison cache format in {path}: {type(obj)}")
            cls._cache_store[path] = obj
        else:
            cls._cache_store[path] = {}
        cls._cache_dirty.setdefault(path, 0)
        return cls._cache_store[path]

    @classmethod
    def _save_cache(cls, path: Path) -> None:
        path = path.resolve()
        cache = cls._cache_store.get(path)
        if cache is None or cls._cache_dirty.get(path, 0) <= 0:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(cache, f)
        tmp_path.replace(path)
        cls._cache_dirty[path] = 0

    def _default_cache_path(self, dataset: str, split: str) -> Path:
        return (
            EXTERNAL_DATA_DIR
            / "live-rerank-cache"
            / f"{self.model_label}_{dataset}_{split}.pkl"
        )

    def _load_query_and_corpus_text(self, dataset: str) -> None:
        dataset_dir = EXTERNAL_DATA_DIR / "beir" / dataset
        corpus_path = dataset_dir / "corpus.jsonl"
        queries_path = dataset_dir / "queries.jsonl"

        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Live FLAN-T5 mode needs document text, but corpus.jsonl is missing: "
                f"{corpus_path}. For TREC-News this corpus is licensed, so the public "
                f"qrels/topics download is not enough."
            )
        if not queries_path.exists():
            raise FileNotFoundError(f"Missing queries file: {queries_path}")

        text_cache_key = str(dataset_dir.resolve())
        cached_text = LiveFlanT5Oracle._text_store.get(text_cache_key)
        if cached_text is not None:
            self._corpus, self._queries = cached_text
            self._validate_allowed_query_text(queries_path)
            return

        self._corpus = {}
        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                doc_id = str(row.get("_id") or "").strip()
                text = _record_text(row)
                if doc_id and text:
                    self._corpus[doc_id] = text

        self._queries = {}
        with queries_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row.get("_id") or "").strip()
                if not qid:
                    continue
                text = _record_text(row)
                source_doc_id = str(row.get("doc_id") or row.get("docid") or "").strip()
                if not text and source_doc_id:
                    text = self._corpus.get(source_doc_id, "")
                if text:
                    self._queries[qid] = text

        LiveFlanT5Oracle._text_store[text_cache_key] = (self._corpus, self._queries)
        self._validate_allowed_query_text(queries_path)

    def _validate_allowed_query_text(self, queries_path: Path) -> None:
        if self._allowed_qids:
            missing = sorted(qid for qid in self._allowed_qids if qid not in self._queries)
            if missing:
                raise FileNotFoundError(
                    f"Missing query text for {len(missing)} qids in {queries_path}; "
                    f"first missing qids: {missing[:5]}"
                )

    def _ensure_task(self) -> tuple[str, list[str]]:
        if self.current_task is None:
            raise RuntimeError("No current task set on live FLAN-T5 oracle.")
        return self.current_task.query_id, self.current_task.candidate_ids

    def _compare_entry(self, qid: str, doc_a: str, doc_b: str) -> Mapping[str, Any]:
        if self._comparison_cache_file is None or self.cache_path is None:
            raise RuntimeError("Live FLAN-T5 cache not loaded. Call load_dataset() first.")

        key = (str(qid), str(doc_a), str(doc_b))
        cached = self._comparison_cache_file.get(key)
        if cached is not None:
            return cached

        query = self._queries.get(str(qid))
        text_a = self._corpus.get(str(doc_a))
        text_b = self._corpus.get(str(doc_b))
        if query is None:
            raise FileNotFoundError(f"Missing query text for qid={qid!r}")
        if text_a is None or text_b is None:
            missing = [doc for doc, text in ((doc_a, text_a), (doc_b, text_b)) if text is None]
            raise FileNotFoundError(f"Missing corpus text for docs: {missing[:5]}")

        prompt = self._build_prompt(query, text_a, text_b)
        score_a, score_b, prompt_tokens = self._score_prompt(prompt)
        entry = {
            "model": self.model_name,
            "scores": [("A", score_a), ("B", score_b)],
            "text": prompt,
            "ptks": int(prompt_tokens),
        }
        self._comparison_cache_file[key] = entry

        cache_path = self.cache_path.resolve()
        LiveFlanT5Oracle._cache_dirty[cache_path] = (
            LiveFlanT5Oracle._cache_dirty.get(cache_path, 0) + 1
        )
        if LiveFlanT5Oracle._cache_dirty[cache_path] >= self.cache_flush_interval:
            LiveFlanT5Oracle._save_cache(cache_path)
        return entry

    def _entry_preference(self, entry: Mapping[str, Any]) -> str | None:
        scores = dict(entry.get("scores", []))
        score_a = scores.get("A")
        score_b = scores.get("B")
        if score_a is None or score_b is None or score_a == score_b:
            return None
        return "A" if float(score_a) > float(score_b) else "B"

    def _build_prompt(self, query: str, passage_a: str, passage_b: str) -> str:
        return (
            "Given a search query and two passages, choose the passage that is more "
            "relevant to the query.\n\n"
            f"Query:\n{query}\n\n"
            f"Passage A:\n{passage_a}\n\n"
            f"Passage B:\n{passage_b}\n\n"
            "Answer with only A or B."
        )

    def _ensure_model(self) -> tuple[Any, Any, str, int, int]:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        cache_key = (self.model_name, device)
        cached = LiveFlanT5Oracle._model_cache.get(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Loading FLAN-T5 model for live oracle: {self.model_name} on {device}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model.to(device)
        model.eval()

        token_a = tokenizer("A", add_special_tokens=False).input_ids[0]
        token_b = tokenizer("B", add_special_tokens=False).input_ids[0]
        cached = (tokenizer, model, device, int(token_a), int(token_b))
        LiveFlanT5Oracle._model_cache[cache_key] = cached
        return cached

    def _score_prompt(self, prompt: str) -> tuple[float, float, int]:
        import torch

        tokenizer, model, device, token_a, token_b = self._ensure_model()
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        inputs = {k: v.to(device) for k, v in inputs.items()}

        decoder_start_token_id = model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.pad_token_id
        decoder_input_ids = torch.tensor([[decoder_start_token_id]], device=device)

        with torch.inference_mode():
            outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

        return float(log_probs[token_a].item()), float(log_probs[token_b].item()), prompt_tokens


class LiveFlanT5BidirectionalOracle(LiveFlanT5Oracle):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(cache_comparisons=True, **kwargs)
        self.name = "Live FLAN-T5 (Bidirectional)"

    def sample_lt(self, i: int, j: int) -> bool:
        qid, candidates = self._ensure_task()
        doc_i, doc_j = candidates[i], candidates[j]
        forward_pref = self._entry_preference(self._compare_entry(qid, doc_i, doc_j))
        reverse_pref = self._entry_preference(self._compare_entry(qid, doc_j, doc_i))
        if forward_pref is None or reverse_pref is None:
            return False
        return forward_pref == "B" and reverse_pref == "A"


class LiveFlanT5SamplingOracle(LiveFlanT5Oracle):
    def __init__(self, *, seed: int | None = None, **kwargs: Any) -> None:
        super().__init__(cache_comparisons=False, **kwargs)
        self.name = "Live FLAN-T5 (Sampling)"
        self._rng = random.Random(seed)

    def set_seed(self, seed: int | None) -> None:
        super().set_seed(seed)
        self._rng = random.Random(seed)

    def sample_lt(self, i: int, j: int) -> bool:
        qid, candidates = self._ensure_task()
        doc_i, doc_j = candidates[i], candidates[j]
        use_forward = self._rng.random() < 0.5
        if use_forward:
            pref = self._entry_preference(self._compare_entry(qid, doc_i, doc_j))
            return pref == "B"
        pref = self._entry_preference(self._compare_entry(qid, doc_j, doc_i))
        return pref == "A"
