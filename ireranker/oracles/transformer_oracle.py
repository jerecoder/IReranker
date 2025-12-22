"""Oracle that performs live pairwise comparisons using transformer models."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from ireranker.config import logger
from ireranker.oracles.oracle import MatrixKey, Oracle


class TransformerOracle(Oracle):
    """Oracle that generates pairwise comparisons by prompting a transformer model.

    This oracle loads a HuggingFace transformer model and performs live inference
    to compare document pairs. Results are cached in a matrix that can be saved
    to disk in the same format as pre-computed matrices.

    Usage:
        oracle = TransformerOracle(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            device="cuda",
            quantization="8bit"
        )
        oracle.load_dataset("scifact", split="test")
        # Use with rankers...
        oracle.save_matrix("output/scifact_llama.pkl")
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        quantization: Optional[str] = None,
        comparison_limit: int | None = None,
        comparison_limit_per_task: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the TransformerOracle.

        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            device: Device to run model on ("cuda", "cpu", or "auto")
            quantization: Quantization mode ("4bit", "8bit", or None)
            comparison_limit: Optional limit on number of comparisons
            comparison_limit_per_task: Whether comparison limit applies per task
            cache_dir: Optional directory for HuggingFace cache
        """
        super().__init__(
            comparison_limit=comparison_limit,
            comparison_limit_per_task=comparison_limit_per_task,
        )

        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.cache_dir = cache_dir

        # Model components (loaded lazily)
        self._model = None
        self._tokenizer = None

        # Dataset state
        self._queries: Dict[str, str] = {}
        self._corpus: Dict[str, str] = {}
        self._dataset_name: Optional[str] = None

        # Matrix cache: (query_id, doc_a_id, doc_b_id) -> comparison result
        self._matrix: Dict[MatrixKey, Mapping[str, Any]] = {}

        # Checkpoint counter for periodic saves
        self._comparison_count_since_checkpoint = 0
        self._checkpoint_interval = 1000  # Save checkpoint every N comparisons
        self._checkpoint_dir: Optional[Path] = None

        # Enable comparison caching by default
        self.enable_cache(True)

    def _load_model(self):
        """Lazily load the transformer model and tokenizer."""
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info(f"Loading model: {self.model_name}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for TransformerOracle. "
                "Install with: pip install transformers torch accelerate"
            ) from e

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )

        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Configure model loading based on quantization
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "torch_dtype": torch.float16,
        }

        if self.quantization == "8bit":
            try:
                import bitsandbytes  # noqa: F401

                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
                logger.info("Using 8-bit quantization")
            except ImportError:
                logger.warning("bitsandbytes not available, loading without quantization")
                model_kwargs["device_map"] = self.device

        elif self.quantization == "4bit":
            try:
                import bitsandbytes  # noqa: F401

                model_kwargs["load_in_4bit"] = True
                model_kwargs["device_map"] = "auto"
                logger.info("Using 4-bit quantization")
            except ImportError:
                logger.warning("bitsandbytes not available, loading without quantization")
                model_kwargs["device_map"] = self.device

        else:
            # No quantization
            model_kwargs["device_map"] = self.device

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        self._model.eval()
        logger.info(f"Model loaded successfully on {self.device}")

    def load_dataset(
        self,
        dataset: str,
        *,
        split: str = "test",
        query_ids: Optional[Iterable[str]] = None,
        matrix_model: Optional[str] = None,
    ) -> None:
        """Load BEIR dataset queries and corpus for comparison generation.

        Args:
            dataset: BEIR dataset name (e.g., "scifact", "dl-2019")
            split: Dataset split (usually "test")
            query_ids: Optional subset of query IDs to use
            matrix_model: Ignored (for API compatibility)
        """
        from ireranker.config import EXTERNAL_DATA_DIR

        # Find BEIR dataset path
        beir_path = EXTERNAL_DATA_DIR / "beir" / dataset.lower()

        if not beir_path.exists():
            raise FileNotFoundError(
                f"BEIR dataset not found: {beir_path}. "
                f"Download it first using the data loader."
            )

        # Load queries
        queries_file = beir_path / "queries.jsonl"
        if not queries_file.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_file}")

        logger.info(f"Loading queries from {queries_file}")
        self._queries = {}
        with open(queries_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                qid = obj["_id"]
                text = obj["text"]
                self._queries[qid] = text

        # Filter by query_ids if provided
        if query_ids is not None:
            allowed = {str(qid) for qid in query_ids}
            self._queries = {qid: text for qid, text in self._queries.items() if qid in allowed}

        logger.info(f"Loaded {len(self._queries)} queries")

        # Load corpus
        corpus_file = beir_path / "corpus.jsonl"
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

        logger.info(f"Loading corpus from {corpus_file}")
        self._corpus = {}
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                doc_id = obj["_id"]
                # Combine title and text for full document content
                title = obj.get("title", "")
                text = obj.get("text", "")
                if title:
                    full_text = f"{title}. {text}"
                else:
                    full_text = text
                self._corpus[doc_id] = full_text

        logger.info(f"Loaded {len(self._corpus)} documents")

        self._dataset_name = dataset
        self._matrix.clear()
        self.current_task = None

    def _create_comparison_prompt(
        self, query: str, doc_a: str, doc_b: str, max_doc_length: int = 512
    ) -> str:
        """Create a prompt for pairwise comparison.

        Args:
            query: Query text
            doc_a: First document text
            doc_b: Second document text
            max_doc_length: Maximum character length for each document

        Returns:
            Formatted prompt string
        """
        # Truncate documents if too long
        if len(doc_a) > max_doc_length:
            doc_a = doc_a[:max_doc_length] + "..."
        if len(doc_b) > max_doc_length:
            doc_b = doc_b[:max_doc_length] + "..."

        prompt = f"""Given a query and two passages, determine which passage is more relevant to the query.

Query: {query}

Passage A: {doc_a}

Passage B: {doc_b}

Which passage is more relevant to the query? Answer only with 'A' or 'B'.

Answer:"""

        return prompt

    def _prompt_model(self, query: str, doc_a: str, doc_b: str) -> Tuple[str, float, float]:
        """Prompt the model to compare two documents.

        Args:
            query: Query text
            doc_a: First document text
            doc_b: Second document text

        Returns:
            Tuple of (preference ["A" or "B"], score_a, score_b)
        """
        self._load_model()

        prompt = self._create_comparison_prompt(query, doc_a, doc_b)

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        # Move to device
        if self.device not in ["auto", "balanced"]:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        import torch

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,  # Deterministic
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode response
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer (should be after "Answer:")
        answer = response.split("Answer:")[-1].strip()

        # Parse preference
        if "A" in answer.upper() and "B" not in answer.upper():
            preference = "A"
            score_a, score_b = 1.0, 0.0
        elif "B" in answer.upper() and "A" not in answer.upper():
            preference = "B"
            score_a, score_b = 0.0, 1.0
        else:
            # Ambiguous or invalid response, default to A
            logger.warning(f"Ambiguous model response: {answer!r}, defaulting to A")
            preference = "A"
            score_a, score_b = 0.5, 0.5

        return preference, score_a, score_b

    def sample_lt(self, i: int, j: int) -> bool:
        """Return True when doc at index i should be ranked after doc at index j.

        This performs the actual model comparison and caches the result.
        """
        if self.current_task is None:
            return False

        qid = self.current_task.query_id
        doc_a_id = self.current_task.candidate_ids[i]
        doc_b_id = self.current_task.candidate_ids[j]

        # Check cache first
        matrix_key = (qid, doc_a_id, doc_b_id)
        if matrix_key in self._matrix:
            entry = self._matrix[matrix_key]
            preference = entry.get("text", "").replace("Passage ", "").strip()
            return preference == "B"

        # Reverse key might exist
        reverse_key = (qid, doc_b_id, doc_a_id)
        if reverse_key in self._matrix:
            entry = self._matrix[reverse_key]
            preference = entry.get("text", "").replace("Passage ", "").strip()
            # If reverse says B, then forward would be A, so i < j is False
            # If reverse says A, then forward would be B, so i < j is True
            return preference == "A"

        # Not cached, need to query model
        query_text = self._queries.get(qid, "")
        doc_a_text = self._corpus.get(doc_a_id, "")
        doc_b_text = self._corpus.get(doc_b_id, "")

        if not query_text or not doc_a_text or not doc_b_text:
            logger.warning(
                f"Missing text for comparison: qid={qid}, " f"doc_a={doc_a_id}, doc_b={doc_b_id}"
            )
            return False

        # Perform comparison
        preference, score_a, score_b = self._prompt_model(query_text, doc_a_text, doc_b_text)

        # Store in matrix (both forward and reverse)
        forward_entry = {
            "text": f"Passage {preference}",
            "scores": [("A", score_a), ("B", score_b)],
        }

        reverse_preference = "B" if preference == "A" else "A"
        reverse_entry = {
            "text": f"Passage {reverse_preference}",
            "scores": [("A", score_b), ("B", score_a)],
        }

        self._matrix[matrix_key] = forward_entry
        self._matrix[reverse_key] = reverse_entry

        # Checkpoint management
        self._comparison_count_since_checkpoint += 1
        if (
            self._checkpoint_dir is not None
            and self._comparison_count_since_checkpoint >= self._checkpoint_interval
        ):
            self._save_checkpoint()
            self._comparison_count_since_checkpoint = 0

        return preference == "B"

    def enable_checkpoints(self, checkpoint_dir: str | Path, interval: int = 1000) -> None:
        """Enable periodic checkpoint saving.

        Args:
            checkpoint_dir: Directory to save checkpoints
            interval: Number of comparisons between checkpoints
        """
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_interval = interval
        logger.info(f"Checkpoints enabled: {self._checkpoint_dir} (interval: {interval})")

    def _save_checkpoint(self) -> None:
        """Save current matrix as a checkpoint."""
        if self._checkpoint_dir is None:
            return

        checkpoint_file = (
            self._checkpoint_dir / f"{self._dataset_name}_checkpoint_{len(self._matrix)}.pkl"
        )

        with open(checkpoint_file, "wb") as f:
            pickle.dump(self._matrix, f)

        logger.info(f"Checkpoint saved: {checkpoint_file} ({len(self._matrix)} entries)")

    def save_matrix(self, output_path: str | Path) -> None:
        """Save the generated matrix to disk in PKL format.

        The saved matrix is compatible with MatrixOracle and can be loaded
        using the standard matrix loading utilities.

        Args:
            output_path: Path to save the matrix file (.pkl)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(self._matrix, f)

        logger.info(
            f"Matrix saved: {output_path} "
            f"({len(self._matrix)} entries, "
            f"{len(self._queries)} queries, "
            f"{self._comparisons} comparisons)"
        )

    def load_matrix(self, matrix_path: str | Path) -> None:
        """Load a previously saved matrix to resume generation.

        Args:
            matrix_path: Path to the saved matrix file (.pkl)
        """
        matrix_path = Path(matrix_path)

        if not matrix_path.exists():
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

        with open(matrix_path, "rb") as f:
            self._matrix = pickle.load(f)

        logger.info(f"Matrix loaded: {matrix_path} ({len(self._matrix)} entries)")
