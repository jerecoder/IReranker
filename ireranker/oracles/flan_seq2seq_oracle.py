"""Live oracle for Flan (seq2seq) models with on-demand comparisons."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from ireranker.config import EXTERNAL_DATA_DIR, logger
from ireranker.oracles.oracle import MatrixKey, Oracle


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
        self._matrix: Dict[MatrixKey, Mapping[str, Any]] = {}
        self._dataset: Optional[str] = None

        self.enable_cache(True)

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
        self._matrix.clear()
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

    # ---- core comparison logic -------------------------------------------------
    def _compare_pair(self, qid: str, doc_a_id: str, doc_b_id: str) -> Mapping[str, Any]:
        self._ensure_model()
        query = self._queries.get(qid, "")
        doc_a = self._corpus.get(doc_a_id, "")
        doc_b = self._corpus.get(doc_b_id, "")
        if len(doc_a) > self.max_doc_len:
            doc_a = doc_a[: self.max_doc_len] + "..."
        if len(doc_b) > self.max_doc_len:
            doc_b = doc_b[: self.max_doc_len] + "..."

        prompt = self.prompt_template.format(query=query, doc_a=doc_a, doc_b=doc_b)
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

    def _ensure_pair(self, qid: str, doc_a_id: str, doc_b_id: str) -> Mapping[str, Any]:
        key = (qid, doc_a_id, doc_b_id)
        if key not in self._matrix:
            meta = self._compare_pair(qid, doc_a_id, doc_b_id)
            pref = meta.get("winner", "A")
            score_a, score_b = meta["scores"][0][1], meta["scores"][1][1]
            self._matrix[key] = meta
            rev_pref = "B" if pref == "A" else "A"
            self._matrix[(qid, doc_b_id, doc_a_id)] = {
                **meta,
                "winner": rev_pref,
                "scores": [("A", score_b), ("B", score_a)],
                "text": f"Passage {rev_pref}",
            }
        return self._matrix[key]

    def sample_lt(self, i: int, j: int) -> bool:
        if self.current_task is None:
            return False
        qid = self.current_task.query_id
        doc_a = self.current_task.candidate_ids[i]
        doc_b = self.current_task.candidate_ids[j]
        entry = self._ensure_pair(qid, doc_a, doc_b)
        pref = entry.get("winner")
        return pref == "B"

    # ---- persistence helpers ---------------------------------------------------
    def save_matrix(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._matrix, f)
        logger.info(f"Saved matrix with {len(self._matrix)} entries to {path}")

    def load_matrix(self, path: Path) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            self._matrix = pickle.load(f)
        logger.info(f"Loaded matrix with {len(self._matrix)} entries from {path}")
