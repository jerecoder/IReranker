"""Live oracle for Flan (seq2seq) models with on-demand comparisons."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from ireranker.config import EXTERNAL_DATA_DIR, logger
from ireranker.oracles.oracle import BudgetExceeded, MatrixKey, Oracle


@dataclass
class _PendingComparison:
    event: threading.Event
    error: Exception | None = None
    want_bidirectional: bool = False


class FlanComparisonStore:
    """Thread-safe store for Flan comparison entries and in-flight requests."""

    def __init__(self) -> None:
        self._matrix: Dict[MatrixKey, Mapping[str, Any]] = {}
        self._pending: Dict[MatrixKey, _PendingComparison] = {}
        self._lock = threading.Lock()
        self._signature: tuple[str, tuple[str, ...] | None] | None = None
        self._comparison_count = 0

    @property
    def matrix(self) -> Dict[MatrixKey, Mapping[str, Any]]:
        return self._matrix

    @property
    def comparisons(self) -> int:
        with self._lock:
            return self._comparison_count

    def snapshot(self) -> Dict[MatrixKey, Mapping[str, Any]]:
        with self._lock:
            return dict(self._matrix)

    def replace_matrix(self, matrix: Mapping[MatrixKey, Mapping[str, Any]]) -> None:
        with self._lock:
            self._matrix.clear()
            self._matrix.update(matrix)
            self._pending.clear()

    def set_comparisons(self, count: int) -> None:
        with self._lock:
            self._comparison_count = count

    def configure(self, dataset: str, query_ids: Optional[Iterable[str]] = None) -> None:
        sig = (
            str(dataset).lower().strip(),
            tuple(sorted(str(qid) for qid in query_ids)) if query_ids is not None else None,
        )
        with self._lock:
            if self._signature != sig:
                self._matrix.clear()
                self._pending.clear()
                self._comparison_count = 0
                self._signature = sig

    def clear(self) -> None:
        with self._lock:
            self._matrix.clear()
            self._pending.clear()
            self._comparison_count = 0
            self._signature = None

    def get(self, key: MatrixKey) -> Optional[Mapping[str, Any]]:
        with self._lock:
            return self._matrix.get(key)

    def reserve(
        self,
        key: MatrixKey,
        *,
        reverse_key: Optional[MatrixKey] = None,
        bidirectional: bool = False,
    ) -> tuple[_PendingComparison, bool]:
        with self._lock:
            pending = self._pending.get(key)
            if pending is None and reverse_key is not None:
                pending = self._pending.get(reverse_key)
            if pending is not None:
                if bidirectional:
                    pending.want_bidirectional = True
                    if reverse_key is not None:
                        self._pending[reverse_key] = pending
                        self._pending[key] = pending
                return pending, False

            pending = _PendingComparison(event=threading.Event(), want_bidirectional=bidirectional)
            self._pending[key] = pending
            if reverse_key is not None and bidirectional:
                self._pending[reverse_key] = pending
            return pending, True

    def fulfill(
        self,
        key: MatrixKey,
        entry: Mapping[str, Any],
        *,
        reverse_key: Optional[MatrixKey] = None,
        reverse_entry: Optional[Mapping[str, Any]] = None,
    ) -> None:
        pending = None
        with self._lock:
            self._matrix[key] = entry
            if reverse_key is not None and reverse_entry is not None:
                self._matrix[reverse_key] = reverse_entry
            pending = self._pending.pop(key, None)
            if reverse_key is not None:
                self._pending.pop(reverse_key, None)
        if pending is not None:
            pending.event.set()

    def fail(
        self,
        key: MatrixKey,
        error: Exception,
        *,
        reverse_key: Optional[MatrixKey] = None,
    ) -> None:
        pending = None
        with self._lock:
            pending = self._pending.pop(key, None)
            if reverse_key is not None:
                self._pending.pop(reverse_key, None)
        if pending is not None:
            pending.error = error
            pending.event.set()

    def increment_comparisons(self, count: int = 1) -> None:
        with self._lock:
            self._comparison_count += count


@dataclass
class _BatchRequest:
    qid: str
    doc_a_id: str
    doc_b_id: str
    prompt: str
    key: MatrixKey
    reverse_key: MatrixKey
    pending: _PendingComparison
    store: FlanComparisonStore


class FlanBatcher:
    """Background batcher that aggregates Flan comparisons into larger GPU batches."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        quantization: Optional[str] = "8bit",
        max_prompt_len: int = 2048,
        max_new_tokens: int = 4,
        cache_dir: Optional[str] = None,
        batch_size: int = 128,
        max_wait_ms: int = 5,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.max_prompt_len = max_prompt_len
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        self.batch_size = max(1, batch_size)
        self.max_wait_ms = max(1, max_wait_ms)

        self._queue: queue.Queue[_BatchRequest] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._model = None
        self._tokenizer = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, request: _BatchRequest) -> None:
        self.start()
        self._queue.put(request)

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info(f"Loading Flan model: {self.model_name} (quant={self.quantization})")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        kwargs: Dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "device_map": "auto" if self.device == "cuda" else {"": "cpu"},
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        if self.quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **kwargs)
        self._model.eval()
        logger.info("Model loaded")

    def _target_device(self) -> str:
        if self._model is None:
            return self.device
        device_map = getattr(self._model, "hf_device_map", None)
        if device_map:
            dev = next(iter(device_map.values()))
            if isinstance(dev, int):
                return f"cuda:{dev}" if torch.cuda.is_available() else "cpu"
            if isinstance(dev, str):
                return dev
            if hasattr(dev, "type"):
                return dev.type
        module_device = getattr(self._model, "device", None)
        if module_device:
            return str(module_device)
        return self.device

    def _invert_entry(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        pref = entry.get("winner", "A")
        rev_pref = "B" if pref == "A" else "A"
        score_a = None
        score_b = None
        raw_scores = entry.get("scores")
        if isinstance(raw_scores, Mapping):
            score_a = raw_scores.get("A")
            score_b = raw_scores.get("B")
        elif isinstance(raw_scores, (list, tuple)):
            for item in raw_scores:
                if isinstance(item, tuple) and len(item) == 2:
                    label, value = item
                    if label == "A":
                        score_a = value
                    elif label == "B":
                        score_b = value
        inverted = dict(entry)
        inverted["winner"] = rev_pref
        if score_a is not None and score_b is not None:
            inverted["scores"] = [("A", score_b), ("B", score_a)]
        inverted["text"] = f"Passage {rev_pref}"
        return inverted

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch: List[_BatchRequest] = [first]
            start = time.time()
            while len(batch) < self.batch_size:
                remaining = (self.max_wait_ms / 1000.0) - (time.time() - start)
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break

            try:
                self._ensure_model()
                prompts = [req.prompt for req in batch]
                inputs = self._tokenizer(
                    prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_prompt_len,
                    padding=True,
                )
                tgt_device = self._target_device()
                inputs = {k: v.to(tgt_device) for k, v in inputs.items()}
                input_lens = inputs["attention_mask"].sum(dim=1).tolist()

                t0 = time.time()
                with torch.no_grad():
                    gen = self._model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self._tokenizer.pad_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                latency_ms = (time.time() - t0) * 1000
                responses = self._tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)
                transition_scores = self._model.compute_transition_scores(
                    gen.sequences, gen.scores, normalize_logits=True
                )

                for idx, req in enumerate(batch):
                    response = responses[idx]
                    upper_resp = response.upper()
                    if "PASSAGE A" in upper_resp or upper_resp.strip().startswith("A"):
                        pref = "A"
                    elif "PASSAGE B" in upper_resp or upper_resp.strip().startswith("B"):
                        pref = "B"
                    else:
                        pref = "A"

                    seq = gen.sequences[idx]
                    input_len = int(input_lens[idx])
                    gen_tokens = seq[input_len:]
                    token_logprobs = [float(s) for s in transition_scores[idx]]

                    score_a, score_b = (1.0, 0.0) if pref == "A" else (0.0, 1.0)
                    meta = {
                        "winner": pref,
                        "response": response,
                        "token_ids": gen_tokens.tolist(),
                        "tokens": self._tokenizer.convert_ids_to_tokens(gen_tokens),
                        "logprobs": token_logprobs,
                        "latency_ms": latency_ms,
                        "tokens_in": input_len,
                        "tokens_out": int(gen_tokens.shape[0]),
                        "text": f"Passage {pref}",
                        "scores": [("A", score_a), ("B", score_b)],
                    }

                    want_bidirectional = req.pending.want_bidirectional
                    reverse_entry = self._invert_entry(meta) if want_bidirectional else None
                    req.store.fulfill(
                        req.key,
                        meta,
                        reverse_key=req.reverse_key if want_bidirectional else None,
                        reverse_entry=reverse_entry,
                    )
            except Exception as exc:
                for req in batch:
                    req.store.fail(req.key, exc, reverse_key=req.reverse_key)


class FlanSeq2SeqOracle(Oracle):
    """On-demand oracle that calls Flan models (seq2seq) to compare pairs."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        quantization: Optional[str] = "8bit",
        prompt_template: str
        = (
            "Given a query {query}, which of the following two passages is more relevant to the query?\n"
            "Passage A: {doc_a}\n"
            "Passage B: {doc_b}\n"
            "Output Passage A or Passage B:"
        ),
        max_prompt_len: int = 2048,
        max_new_tokens: int = 4,
        max_doc_len: int = 400,
        cache_dir: Optional[str] = None,
        comparison_limit: int | None = None,
        comparison_limit_per_task: bool = False,
        bidirectional: bool = True,
        shared_store: FlanComparisonStore | None = None,
        batcher: FlanBatcher | None = None,
    ):
        super().__init__(
            comparison_limit=comparison_limit,
            comparison_limit_per_task=comparison_limit_per_task,
        )
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.prompt_template = prompt_template
        self.max_prompt_len = max_prompt_len
        self.max_new_tokens = max_new_tokens
        self.max_doc_len = max_doc_len
        self.cache_dir = cache_dir

        self._model = None
        self._tokenizer = None
        self._queries: Dict[str, str] = {}
        self._corpus: Dict[str, str] = {}
        self._store = shared_store or FlanComparisonStore()
        self._batcher = batcher
        self._matrix: Dict[MatrixKey, Mapping[str, Any]] = self._store.matrix
        self._dataset: Optional[str] = None
        self._comparison_count: int = 0
        self._bidirectional = bidirectional
        self._comparison_weight = 2 if bidirectional else 1

        self.enable_cache(False)

    # ---- dataset/model helpers -------------------------------------------------
    def load_dataset(
        self,
        dataset: str,
        *,
        split: str = "test",
        query_ids: Optional[Iterable[str]] = None,
        matrix_model: Optional[str] = None,  # unused; for API compatibility
    ) -> None:
        beir_path = EXTERNAL_DATA_DIR / "beir" / dataset.lower()
        queries_file = beir_path / f"queries.jsonl"
        corpus_file = beir_path / f"corpus.jsonl"
        if not queries_file.exists() or not corpus_file.exists():
            raise FileNotFoundError(f"BEIR data missing at {beir_path}")

        logger.info(f"Loading BEIR dataset: {dataset} ({split})")
        self._queries.clear()
        with open(queries_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                qid = obj["_id"]
                if query_ids is not None and str(qid) not in {str(q) for q in query_ids}:
                    continue
                self._queries[qid] = obj["text"]

        self._corpus.clear()
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                doc_id = obj["_id"]
                title = obj.get("title", "")
                text = obj.get("text", "")
                self._corpus[doc_id] = f"{title}. {text}" if title else text

        logger.info(f"Loaded {len(self._queries)} queries, {len(self._corpus)} docs")
        self._dataset = dataset
        self._store.configure(dataset, query_ids=query_ids)
        self._comparison_count = self._store.comparisons
        self.reset_comparisons()

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info(f"Loading Flan model: {self.model_name} (quant={self.quantization})")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        kwargs: Dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }

        # Only use device_map for CUDA (requires accelerate)
        if self.device == "cuda":
            kwargs["device_map"] = "auto"

        if self.quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **kwargs)

        # For CPU, manually move model after loading
        if self.device == "cpu":
            self._model = self._model.to("cpu")

        self._model.eval()
        logger.info("Model loaded")

    def _target_device(self) -> str:
        if self._model is None:
            return self.device
        device_map = getattr(self._model, "hf_device_map", None)
        if device_map:
            dev = next(iter(device_map.values()))
            if isinstance(dev, int):
                return f"cuda:{dev}" if torch.cuda.is_available() else "cpu"
            if isinstance(dev, str):
                return dev
            if hasattr(dev, "type"):
                return dev.type
        module_device = getattr(self._model, "device", None)
        if module_device:
            return str(module_device)
        return self.device

    @staticmethod
    def _extract_scores(entry: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        raw_scores = entry.get("scores")
        score_a = None
        score_b = None
        if isinstance(raw_scores, Mapping):
            score_a = raw_scores.get("A")
            score_b = raw_scores.get("B")
        elif isinstance(raw_scores, (list, tuple)):
            for item in raw_scores:
                if isinstance(item, tuple) and len(item) == 2:
                    label, value = item
                    if label == "A":
                        score_a = value
                    elif label == "B":
                        score_b = value
        return score_a, score_b

    def _invert_entry(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        pref = entry.get("winner", "A")
        rev_pref = "B" if pref == "A" else "A"
        score_a, score_b = self._extract_scores(entry)
        inverted = dict(entry)
        inverted["winner"] = rev_pref
        if score_a is not None and score_b is not None:
            inverted["scores"] = [("A", score_b), ("B", score_a)]
        inverted["text"] = f"Passage {rev_pref}"
        return inverted

    def _build_prompt(self, qid: str, doc_a_id: str, doc_b_id: str) -> str:
        query = self._queries.get(qid, "")
        doc_a = self._corpus.get(doc_a_id, "")
        doc_b = self._corpus.get(doc_b_id, "")
        if len(doc_a) > self.max_doc_len:
            doc_a = doc_a[: self.max_doc_len] + "..."
        if len(doc_b) > self.max_doc_len:
            doc_b = doc_b[: self.max_doc_len] + "..."
        return self.prompt_template.format(query=query, doc_a=doc_a, doc_b=doc_b)

    # ---- core comparison logic -------------------------------------------------
    def _compare_pair(self, qid: str, doc_a_id: str, doc_b_id: str) -> Mapping[str, Any]:
        self._ensure_model()
        prompt = self._build_prompt(qid, doc_a_id, doc_b_id)
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_prompt_len
        )
        tgt_device = self._target_device()
        inputs = {k: v.to(tgt_device) for k, v in inputs.items()}

        t0 = time.time()
        gen = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        latency_ms = (time.time() - t0) * 1000

        response = self._tokenizer.decode(gen.sequences[0], skip_special_tokens=True)
        upper_resp = response.upper()
        if "PASSAGE A" in upper_resp or upper_resp.strip().startswith("A"):
            pref = "A"
        elif "PASSAGE B" in upper_resp or upper_resp.strip().startswith("B"):
            pref = "B"
        else:
            pref = "A"

        gen_tokens = gen.sequences[0, inputs["input_ids"].shape[1]:]
        transition_scores = self._model.compute_transition_scores(
            gen.sequences, gen.scores, normalize_logits=True
        )[0]
        token_logprobs = [float(s) for s in transition_scores]

        score_a, score_b = (1.0, 0.0) if pref == "A" else (0.0, 1.0)
        meta = {
            "winner": pref,
            "response": response,
            "token_ids": gen_tokens.tolist(),
            "tokens": self._tokenizer.convert_ids_to_tokens(gen_tokens),
            "logprobs": token_logprobs,
            "latency_ms": latency_ms,
            "tokens_in": int(inputs["input_ids"].shape[1]),
            "tokens_out": int(gen_tokens.shape[0]),
            "text": f"Passage {pref}",
            "scores": [("A", score_a), ("B", score_b)],
        }
        return meta

    def _get_entry_for_pair(
        self, qid: str, doc_a_id: str, doc_b_id: str
    ) -> Tuple[Mapping[str, Any], bool]:
        key = (qid, doc_a_id, doc_b_id)
        reverse_key = (qid, doc_b_id, doc_a_id)
        entry = self._store.get(key)
        if entry is not None:
            return entry, False

        reverse_entry = self._store.get(reverse_key)
        if reverse_entry is not None:
            entry = self._invert_entry(reverse_entry)
            if self._bidirectional:
                self._store.fulfill(key, entry)
            return entry, False

        pending, is_new = self._store.reserve(
            key, reverse_key=reverse_key, bidirectional=self._bidirectional
        )
        if not is_new:
            pending.event.wait()
            if pending.error:
                raise pending.error
            entry = self._store.get(key)
            if entry is not None:
                return entry, False
            reverse_entry = self._store.get(reverse_key)
            if reverse_entry is not None:
                entry = self._invert_entry(reverse_entry)
                if self._bidirectional:
                    self._store.fulfill(key, entry)
                return entry, False
            raise RuntimeError(f"Missing comparison for {key} after pending completion.")

        try:
            if self._batcher is not None:
                prompt = self._build_prompt(qid, doc_a_id, doc_b_id)
                request = _BatchRequest(
                    qid=qid,
                    doc_a_id=doc_a_id,
                    doc_b_id=doc_b_id,
                    prompt=prompt,
                    key=key,
                    reverse_key=reverse_key,
                    pending=pending,
                    store=self._store,
                )
                self._batcher.enqueue(request)
                pending.event.wait()
                if pending.error:
                    raise pending.error
                entry = self._store.get(key)
                if entry is None:
                    reverse_entry = self._store.get(reverse_key)
                    if reverse_entry is not None:
                        entry = self._invert_entry(reverse_entry)
                        if self._bidirectional:
                            self._store.fulfill(key, entry)
                if entry is None:
                    raise RuntimeError(f"Missing comparison for {key} after batcher completion.")
            else:
                entry = self._compare_pair(qid, doc_a_id, doc_b_id)
                want_bidirectional = pending.want_bidirectional
                reverse_entry = self._invert_entry(entry) if want_bidirectional else None
                self._store.fulfill(
                    key,
                    entry,
                    reverse_key=reverse_key if want_bidirectional else None,
                    reverse_entry=reverse_entry,
                )

            self._store.increment_comparisons(1)
            self._comparison_count = self._store.comparisons
            if self._comparison_count % 10 == 0:
                pref = entry.get("winner", "A")
                logger.info(
                    f"Completed {self._comparison_count} comparisons (latest: query={qid}, winner={pref}, latency={entry['latency_ms']:.1f}ms)"
                )
            return entry, True
        except Exception as exc:
            self._store.fail(key, exc, reverse_key=reverse_key)
            raise

    def sample_lt(self, i: int, j: int) -> bool:
        if self.current_task is None:
            return False
        qid = self.current_task.query_id
        doc_a = self.current_task.candidate_ids[i]
        doc_b = self.current_task.candidate_ids[j]
        entry, _ = self._get_entry_for_pair(qid, doc_a, doc_b)
        pref = entry.get("winner")
        return pref == "B"

    def lt(self, i: int, j: int) -> bool:
        if self.current_task is None:
            return False

        if self.comparison_limit is not None:
            current_comparisons = (
                self._task_comparisons if self.comparison_limit_per_task else self._comparisons
            )
            if current_comparisons >= self.comparison_limit:
                raise BudgetExceeded(f"Comparison limit of {self.comparison_limit} exceeded.")

        self._comparison_calls += 1
        qid = self.current_task.query_id
        doc_a = self.current_task.candidate_ids[i]
        doc_b = self.current_task.candidate_ids[j]
        entry, is_new = self._get_entry_for_pair(qid, doc_a, doc_b)
        if is_new:
            self._comparisons += self._comparison_weight
            self._task_comparisons += self._comparison_weight
        else:
            self._cache_hits += 1
        pref = entry.get("winner")
        return pref == "B"

    # ---- persistence helpers ---------------------------------------------------
    @staticmethod
    def _unique_pair_count(matrix: Mapping[MatrixKey, Any]) -> int:
        seen: set[MatrixKey] = set()
        for qid, doc_a, doc_b in matrix.keys():
            if doc_a <= doc_b:
                seen.add((qid, doc_a, doc_b))
            else:
                seen.add((qid, doc_b, doc_a))
        return len(seen)

    def save_matrix(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = self._store.snapshot()
        with open(path, "wb") as f:
            pickle.dump(matrix, f)
        self._comparison_count = self._store.comparisons
        logger.info(
            f"Saved matrix with {len(matrix)} entries ({self._comparison_count} comparisons) to {path}"
        )

    def load_matrix(self, path: Path) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            matrix = pickle.load(f)
        self._store.replace_matrix(matrix)
        self._store.set_comparisons(self._unique_pair_count(matrix))
        self._comparison_count = self._store.comparisons
        logger.info(f"Loaded matrix with {len(matrix)} entries from {path}")


class DirectionalFlanSeq2SeqOracle(FlanSeq2SeqOracle):
    """Flan oracle that stores one entry per unordered pair (directional)."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("bidirectional", None)
        super().__init__(*args, **kwargs, bidirectional=False)


class BidirectionalFlanSeq2SeqOracle(FlanSeq2SeqOracle):
    """Flan oracle that stores both directions for each pair (bidirectional)."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("bidirectional", None)
        super().__init__(*args, **kwargs, bidirectional=True)
