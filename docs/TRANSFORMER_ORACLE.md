# TransformerOracle: Live Matrix Generation

The `TransformerOracle` enables on-the-fly generation of pairwise comparison matrices using transformer models like Llama. This allows you to generate new matrices without pre-computing them offline.

## Overview

Traditional IReranker workflow:
1. Pre-compute matrices offline (separate pipeline)
2. Load matrices with `MatrixOracle`
3. Run evaluation

New TransformerOracle workflow:
1. Initialize `TransformerOracle` with model
2. Generate matrices during evaluation OR pre-generate and save
3. Use saved matrices with existing evaluation pipeline

## Quick Start

### Option 1: Google Colab (Recommended for Large Models)

1. **Open the Colab notebook**: `notebooks/generate_llama_matrices_colab.ipynb`

2. **Set up runtime**:
   - Runtime → Change runtime type → GPU (V100 or A100 preferred)

3. **Run all cells**:
   - Installs dependencies
   - Loads Llama models with quantization
   - Generates matrices for configured datasets
   - Saves to Google Drive

4. **Download matrices** and place in `data/external/reranking-matrices/llama/`

### Option 2: Local/Server Script

```bash
# Generate matrices for all models and datasets
python scripts/generate_matrices.py

# Generate for specific model
python scripts/generate_matrices.py --model llama-8b

# Generate for specific datasets
python scripts/generate_matrices.py --datasets scifact dl-2019

# Test mode (10 queries only)
python scripts/generate_matrices.py --test

# Resume from checkpoint
python scripts/generate_matrices.py --resume data/checkpoints/llama-8b/scifact_checkpoint_5000.pkl
```

### Option 3: Python API

```python
from ireranker.oracles import TransformerOracle

# Initialize oracle
oracle = TransformerOracle(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    device="cuda",
    quantization="8bit"
)

# Load dataset
oracle.load_dataset("scifact", split="test")

# Enable checkpoints (recommended)
oracle.enable_checkpoints("checkpoints/llama-8b", interval=1000)

# Use with ranker (comparisons are generated automatically)
from ireranker.rankers import BubbleRanker
ranker = BubbleRanker(oracle=oracle)
# ... run evaluation ...

# Save generated matrix
oracle.save_matrix("data/external/reranking-matrices/llama/llama-8b/scifact.pkl")
```

## Configuration

### Model Configuration (`config/llama_models.json`)

```json
{
  "models": {
    "llama-8b": {
      "name": "meta-llama/Llama-3.1-8B-Instruct",
      "quantization": "8bit",
      "min_vram_gb": 12
    },
    "llama-70b": {
      "name": "meta-llama/Llama-3.1-70B-Instruct",
      "quantization": "4bit",
      "min_vram_gb": 16
    }
  }
}
```

### Quantization Options

- **None**: Full precision (fp16) - fastest but highest memory
- **8bit**: ~50% memory reduction, minimal quality loss
- **4bit**: ~75% memory reduction, some quality loss

**Memory Requirements:**

| Model | No Quant | 8-bit | 4-bit |
|-------|----------|-------|-------|
| Llama 8B | 24 GB | 12 GB | 6 GB |
| Llama 70B | 140 GB | 70 GB | 35 GB |
| Llama 405B | 810 GB | 405 GB | 203 GB |

**Recommendations:**
- V100 (16GB): Llama 8B with 8-bit, Llama 70B with 4-bit
- A100 (40GB): Llama 8B no quant, Llama 70B with 4-bit
- A100 (80GB): Llama 70B with 8-bit

## Features

### Checkpointing

Automatic checkpointing prevents data loss on interruptions:

```python
oracle.enable_checkpoints(
    checkpoint_dir="checkpoints/llama-8b",
    interval=1000  # Save every 1000 comparisons
)
```

Checkpoints are saved as:
- `checkpoints/llama-8b/scifact_checkpoint_5000.pkl`
- `checkpoints/llama-8b/scifact_checkpoint_10000.pkl`

Resume from checkpoint:
```python
oracle.load_matrix("checkpoints/llama-8b/scifact_checkpoint_5000.pkl")
# Continue generation...
```

### Caching

TransformerOracle automatically caches comparisons:
- **Forward cache**: (qid, doc_a, doc_b) → result
- **Reverse cache**: (qid, doc_b, doc_a) → flipped result

This means each document pair is only compared once, even if the ranker asks for both directions.

### Prompt Customization

Modify the prompt template in `config/llama_models.json`:

```json
{
  "prompt_template": "Your custom prompt with {query}, {doc_a}, {doc_b}"
}
```

Or override in code:

```python
class CustomTransformerOracle(TransformerOracle):
    def _create_comparison_prompt(self, query, doc_a, doc_b, max_doc_length=512):
        return f"Custom prompt: {query}, {doc_a[:100]}, {doc_b[:100]}"
```

## Matrix Format

Generated matrices are compatible with existing `MatrixOracle`:

```python
{
    ("query_id", "doc_a_id", "doc_b_id"): {
        "text": "Passage A",  # or "Passage B"
        "scores": [("A", 0.95), ("B", 0.05)]
    }
}
```

This format allows seamless integration with:
- `MatrixOracle`
- `BidirectionalMatrixOracle`
- `SamplingMatrixOracle`
- All existing rankers

## Usage with Evaluation Pipeline

### Generate and Save (Recommended)

```python
# 1. Generate matrices
oracle = TransformerOracle("meta-llama/Llama-3.1-8B-Instruct", quantization="8bit")
oracle.load_dataset("scifact")
# ... trigger comparisons via ranker ...
oracle.save_matrix("data/external/reranking-matrices/llama/llama-8b/scifact.pkl")

# 2. Use in evaluation
from ireranker.run_beir_eval import run_beir_eval
run_beir_eval(matrix_models=["llama-8b"], datasets=["scifact"])
```

### Live Generation (Advanced)

```python
# Use TransformerOracle directly in evaluation
from ireranker.rankers import BubbleRanker
from ireranker.types import RankingTask

oracle = TransformerOracle("meta-llama/Llama-3.1-8B-Instruct")
oracle.load_dataset("scifact")
ranker = BubbleRanker(oracle=oracle)

# Process tasks
for task in tasks:
    ranked_indices = ranker.rank(task)
    # ... evaluate ...
```

## Performance Estimates

Based on V100 GPU:

| Model | Comparisons/sec | 100 queries* | 1000 queries* |
|-------|----------------|--------------|---------------|
| Llama 8B (8-bit) | ~5-10 | 1-2 hours | 10-20 hours |
| Llama 70B (4-bit) | ~1-2 | 5-10 hours | 50-100 hours |

*Assuming ~100 documents per query, ~5000 comparisons per query

**Tips for faster generation:**
- Use smaller models (8B vs 70B)
- Use aggressive quantization (4-bit)
- Reduce max_doc_length (512 → 256 chars)
- Process fewer queries for testing
- Use multiple GPUs if available

## Troubleshooting

### Out of Memory

**Solution 1**: Use stronger quantization
```python
oracle = TransformerOracle(..., quantization="4bit")  # instead of "8bit"
```

**Solution 2**: Reduce document length
```python
class ShortDocOracle(TransformerOracle):
    def _create_comparison_prompt(self, query, doc_a, doc_b, max_doc_length=256):
        # Reduced from default 512
        return super()._create_comparison_prompt(query, doc_a, doc_b, max_doc_length)
```

**Solution 3**: Use model offloading
```python
oracle = TransformerOracle(..., device="auto")  # Offload to CPU if needed
```

### Slow Generation

**Solution 1**: Enable checkpoints and run in batches
```python
oracle.enable_checkpoints("checkpoints", interval=500)
# Run for a few hours, then resume later
```

**Solution 2**: Use smaller test set first
```python
oracle.load_dataset("scifact", query_ids=list(queries.keys())[:10])
```

**Solution 3**: Generate matrices overnight/over weekend

### Model Access

For Llama models, you need HuggingFace access:

1. Go to https://huggingface.co/meta-llama
2. Request access to the model
3. Generate access token: Settings → Access Tokens
4. Login: `huggingface-cli login`

## Examples

### Generate for Multiple Datasets

```python
from ireranker.oracles import TransformerOracle

oracle = TransformerOracle(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization="8bit"
)

oracle.enable_checkpoints("checkpoints/llama-8b")

datasets = ["scifact", "nfcorpus", "trec-covid"]
for dataset in datasets:
    oracle.load_dataset(dataset)
    # ... run evaluation ...
    oracle.save_matrix(f"matrices/llama-8b/{dataset}.pkl")
```

### Compare Multiple Models

```bash
# Generate matrices for multiple models
python scripts/generate_matrices.py --datasets scifact

# This creates:
# - data/external/reranking-matrices/llama/llama-8b/scifact.pkl
# - data/external/reranking-matrices/llama/llama-70b/scifact.pkl

# Evaluate all models
make beir-eval ARGS="--matrix-models llama-8b,llama-70b"

# Compare results
python notebooks/beir_results.ipynb
```

## Integration with Existing Code

TransformerOracle is a drop-in replacement for MatrixOracle:

```python
# Old code
from ireranker.oracles import MatrixOracle
oracle = MatrixOracle()
oracle.load_dataset("scifact", matrix_model="flan-t5-large")

# New code (same interface!)
from ireranker.oracles import TransformerOracle
oracle = TransformerOracle("meta-llama/Llama-3.1-8B-Instruct")
oracle.load_dataset("scifact")
```

All existing rankers work without modification:
- BubbleRanker, QuicksortTopKRanker, PRPSortingRanker
- MohajerRanker, PACRanker
- SpectralMLERanker, SlidingWindowRanker
- ... all others

## Next Steps

1. **Test locally**: Run `python scripts/generate_matrices.py --test`
2. **Generate matrices**: Use Colab notebook for GPU access
3. **Evaluate**: Compare Llama results with existing models
4. **Experiment**: Try different prompts, models, quantization levels

## References

- TransformerOracle implementation: `ireranker/oracles/transformer_oracle.py`
- Configuration: `config/llama_models.json`
- Colab notebook: `notebooks/generate_llama_matrices_colab.ipynb`
- Generation script: `scripts/generate_matrices.py`
