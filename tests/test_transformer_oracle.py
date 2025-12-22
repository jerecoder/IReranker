"""Tests for TransformerOracle."""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ireranker.oracles import TransformerOracle
from ireranker.types import RankingTask


class MockModel:
    """Mock transformer model for testing."""

    def eval(self):
        pass

    def generate(self, **kwargs):
        # Mock output tokens that spell "A"
        import torch
        return torch.tensor([[1, 2, 3]])  # Mock token IDs


class MockTokenizer:
    """Mock tokenizer for testing."""

    pad_token = None
    eos_token = "</s>"
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        # Return mock tokenized input
        import torch
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

    def decode(self, tokens, **kwargs):
        # Mock response that prefers A
        return "Query comparison...\n\nAnswer: A"

    @staticmethod
    def from_pretrained(model_name, **kwargs):
        return MockTokenizer()


@pytest.fixture
def temp_beir_dataset(tmp_path):
    """Create a minimal BEIR dataset for testing."""
    dataset_dir = tmp_path / "beir" / "test-dataset"
    dataset_dir.mkdir(parents=True)

    # Create queries.jsonl
    queries = [
        {"_id": "q1", "text": "What is machine learning?"},
        {"_id": "q2", "text": "How does deep learning work?"}
    ]
    with open(dataset_dir / "queries.jsonl", "w") as f:
        for q in queries:
            f.write(f"{q}\n".replace("'", '"'))

    # Create corpus.jsonl
    corpus = [
        {"_id": "d1", "title": "ML Basics", "text": "Machine learning is a subset of AI."},
        {"_id": "d2", "title": "DL Guide", "text": "Deep learning uses neural networks."},
        {"_id": "d3", "title": "AI Overview", "text": "Artificial intelligence mimics human intelligence."}
    ]
    with open(dataset_dir / "corpus.jsonl", "w") as f:
        for doc in corpus:
            f.write(f"{doc}\n".replace("'", '"'))

    return dataset_dir


@patch("ireranker.oracles.transformer_oracle.AutoModelForCausalLM")
@patch("ireranker.oracles.transformer_oracle.AutoTokenizer")
def test_transformer_oracle_init(mock_tokenizer_class, mock_model_class):
    """Test TransformerOracle initialization."""
    oracle = TransformerOracle(
        model_name="test-model",
        device="cpu",
        quantization=None
    )

    assert oracle.model_name == "test-model"
    assert oracle.device == "cpu"
    assert oracle.quantization is None
    assert oracle._model is None  # Lazy loading
    assert oracle._tokenizer is None


@patch("ireranker.oracles.transformer_oracle.AutoModelForCausalLM")
@patch("ireranker.oracles.transformer_oracle.AutoTokenizer")
def test_load_model(mock_tokenizer_class, mock_model_class):
    """Test model loading."""
    mock_tokenizer_class.from_pretrained.return_value = MockTokenizer()
    mock_model_class.from_pretrained.return_value = MockModel()

    oracle = TransformerOracle("test-model", device="cpu")
    oracle._load_model()

    assert oracle._model is not None
    assert oracle._tokenizer is not None
    mock_model_class.from_pretrained.assert_called_once()


def test_load_dataset(temp_beir_dataset, monkeypatch):
    """Test dataset loading."""
    # Mock EXTERNAL_DATA_DIR to point to temp directory
    from ireranker import config
    monkeypatch.setattr(config, "EXTERNAL_DATA_DIR", temp_beir_dataset.parent)

    oracle = TransformerOracle("test-model")
    oracle.load_dataset("test-dataset", split="test")

    assert len(oracle._queries) == 2
    assert len(oracle._corpus) == 3
    assert "q1" in oracle._queries
    assert "d1" in oracle._corpus
    assert oracle._dataset_name == "test-dataset"


def test_create_comparison_prompt():
    """Test prompt creation."""
    oracle = TransformerOracle("test-model")

    prompt = oracle._create_comparison_prompt(
        query="What is AI?",
        doc_a="AI is artificial intelligence.",
        doc_b="Machine learning is a subset of AI.",
        max_doc_length=100
    )

    assert "What is AI?" in prompt
    assert "AI is artificial intelligence." in prompt
    assert "Machine learning is a subset of AI." in prompt
    assert "Passage A:" in prompt
    assert "Passage B:" in prompt
    assert "Answer:" in prompt


def test_create_comparison_prompt_truncation():
    """Test that long documents are truncated."""
    oracle = TransformerOracle("test-model")

    long_doc = "A" * 1000
    prompt = oracle._create_comparison_prompt(
        query="Test query",
        doc_a=long_doc,
        doc_b="Short doc",
        max_doc_length=100
    )

    # Check that doc_a is truncated
    assert "..." in prompt
    assert len(prompt) < len(long_doc) + 500


@patch("ireranker.oracles.transformer_oracle.AutoModelForCausalLM")
@patch("ireranker.oracles.transformer_oracle.AutoTokenizer")
def test_prompt_model(mock_tokenizer_class, mock_model_class):
    """Test model prompting."""
    mock_tokenizer = MockTokenizer()
    mock_model = MockModel()

    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model_class.from_pretrained.return_value = mock_model

    oracle = TransformerOracle("test-model", device="cpu")

    preference, score_a, score_b = oracle._prompt_model(
        query="Test query",
        doc_a="Document A",
        doc_b="Document B"
    )

    assert preference in ["A", "B"]
    assert 0 <= score_a <= 1
    assert 0 <= score_b <= 1
    assert score_a + score_b == 1.0


def test_save_and_load_matrix(temp_beir_dataset, monkeypatch):
    """Test matrix saving and loading."""
    from ireranker import config
    monkeypatch.setattr(config, "EXTERNAL_DATA_DIR", temp_beir_dataset.parent)

    oracle = TransformerOracle("test-model")
    oracle.load_dataset("test-dataset")

    # Manually add some entries to the matrix
    oracle._matrix[("q1", "d1", "d2")] = {
        "text": "Passage A",
        "scores": [("A", 0.9), ("B", 0.1)]
    }

    # Save matrix
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        matrix_path = Path(f.name)

    oracle.save_matrix(matrix_path)

    # Load matrix
    oracle2 = TransformerOracle("test-model")
    oracle2.load_dataset("test-dataset")
    oracle2.load_matrix(matrix_path)

    assert len(oracle2._matrix) == 1
    assert ("q1", "d1", "d2") in oracle2._matrix

    # Clean up
    matrix_path.unlink()


@patch("ireranker.oracles.transformer_oracle.AutoModelForCausalLM")
@patch("ireranker.oracles.transformer_oracle.AutoTokenizer")
def test_sample_lt_with_cache(mock_tokenizer_class, mock_model_class, temp_beir_dataset, monkeypatch):
    """Test sample_lt with caching."""
    from ireranker import config
    monkeypatch.setattr(config, "EXTERNAL_DATA_DIR", temp_beir_dataset.parent)

    mock_tokenizer = MockTokenizer()
    mock_model = MockModel()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model_class.from_pretrained.return_value = mock_model

    oracle = TransformerOracle("test-model", device="cpu")
    oracle.load_dataset("test-dataset")

    # Create a task
    task = RankingTask(
        query_id="q1",
        candidate_ids=["d1", "d2", "d3"],
        y_true=[1.0, 0.5, 0.0]
    )
    oracle.set_task(task)

    # First comparison should call model
    result1 = oracle.sample_lt(0, 1)
    assert isinstance(result1, bool)

    # Matrix should be populated
    assert len(oracle._matrix) > 0

    # Second call with same indices should use cache
    result2 = oracle.sample_lt(0, 1)
    assert result1 == result2


def test_checkpoint_functionality(temp_beir_dataset, monkeypatch):
    """Test checkpoint saving."""
    from ireranker import config
    monkeypatch.setattr(config, "EXTERNAL_DATA_DIR", temp_beir_dataset.parent)

    oracle = TransformerOracle("test-model")
    oracle.load_dataset("test-dataset")

    # Enable checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        oracle.enable_checkpoints(checkpoint_dir, interval=2)

        # Add entries
        oracle._matrix[("q1", "d1", "d2")] = {"text": "Passage A", "scores": [("A", 1.0), ("B", 0.0)]}
        oracle._comparison_count_since_checkpoint = 2

        # Trigger checkpoint
        oracle._save_checkpoint()

        # Check checkpoint was created
        checkpoints = list(checkpoint_dir.glob("*.pkl"))
        assert len(checkpoints) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
