#!/usr/bin/env python3
"""Experiment: Measure inference time for PRP (Pairwise Ranking Prompting) comparisons.

This script measures the average forward pass time for a FLAN T5 model performing
pairwise document comparisons. Use this to estimate total experiment duration.

Usage:
    # Quick test (few passes, small prompt)
    python experiments/inference_time.py --num-passes 10 --prompt-tokens 100

    # Realistic measurement (default settings)
    python experiments/inference_time.py

    # Custom configuration
    python experiments/inference_time.py --model google/flan-t5-xl --prompt-tokens 512 --num-passes 100

    # CPU mode (no GPU)
    python experiments/inference_time.py --device cpu --num-passes 10
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

# PRP prompt template (same as FlanSeq2SeqOracle)
PROMPT_TEMPLATE = (
    "Given a query {query}, which of the following two passages is more relevant to the query?\n"
    "Passage A: {doc_a}\n"
    "Passage B: {doc_b}\n"
    "Output Passage A or Passage B:"
)


def generate_dummy_prompt(tokenizer, target_tokens: int) -> str:
    """Generate a dummy prompt that tokenizes to approximately target_tokens.
    
    Uses the same prompt template as FlanSeq2SeqOracle to simulate realistic
    PRP comparison prompts.
    """
    # Start with template overhead
    base_prompt = PROMPT_TEMPLATE.format(
        query="dummy query",
        doc_a="",
        doc_b=""
    )
    base_tokens = len(tokenizer.encode(base_prompt, add_special_tokens=True))
    
    # Calculate remaining tokens to fill
    remaining_tokens = max(0, target_tokens - base_tokens)
    tokens_per_field = remaining_tokens // 3  # Split among query, doc_a, doc_b
    
    # Generate dummy text (simple repeated word pattern)
    # Each word typically tokenizes to 1-2 tokens
    dummy_words = " ".join(["word"] * (tokens_per_field * 2))
    
    # Build prompt with dummy content
    prompt = PROMPT_TEMPLATE.format(
        query=dummy_words[:tokens_per_field * 5],  # ~tokens_per_field tokens
        doc_a=dummy_words[:tokens_per_field * 5],
        doc_b=dummy_words[:tokens_per_field * 5]
    )
    
    # Verify and adjust token count
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
    
    # If we're way off, adjust by adding/removing words
    iteration = 0
    while abs(actual_tokens - target_tokens) > 10 and iteration < 20:
        if actual_tokens < target_tokens:
            # Add more content
            extra = " word" * ((target_tokens - actual_tokens) // 2)
            prompt = prompt.replace("Passage B:", f"Passage B:{extra}")
        else:
            # Trim content - reduce doc_b
            words = prompt.split()
            trim_amount = min(len(words) - 20, (actual_tokens - target_tokens) * 2)
            prompt = " ".join(words[:-trim_amount]) if trim_amount > 0 else prompt
        
        actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
        iteration += 1
    
    return prompt


def load_model(
    model_name: str,
    device: str,
    quantization: Optional[str],
    cache_dir: Optional[str] = None
) -> tuple:
    """Load model and tokenizer following the same pattern as FlanSeq2SeqOracle."""
    print(f"Loading model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Quantization: {quantization or 'none'}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    kwargs: Dict[str, Any] = {
        "cache_dir": cache_dir,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    
    # Only use device_map for CUDA
    if device == "cuda":
        kwargs["device_map"] = "auto"
    
    if quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
    
    # For CPU, manually move model after loading
    if device == "cpu":
        model = model.to("cpu")
    
    model.eval()
    print("Model loaded successfully\n")
    
    return model, tokenizer


def get_target_device(model, default_device: str) -> str:
    """Get the actual device the model is on."""
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        dev = next(iter(device_map.values()))
        if isinstance(dev, int):
            return f"cuda:{dev}" if torch.cuda.is_available() else "cpu"
        if isinstance(dev, str):
            return dev
        if hasattr(dev, "type"):
            return dev.type
    module_device = getattr(model, "device", None)
    if module_device:
        return str(module_device)
    return default_device


def measure_inference_time(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    n_passes: int,
    n_warmup: int
) -> List[float]:
    """Run inference N times and return timing for each pass (in milliseconds).
    
    Tokenization is done once before timing. Warmup passes are excluded from results.
    Uses torch.cuda.synchronize() for accurate GPU timing.
    """
    # Tokenize once (outside timing loop)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    target_device = get_target_device(model, device)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    
    input_token_count = inputs["input_ids"].shape[1]
    print(f"Input tokens: {input_token_count}")
    print(f"Running {n_warmup} warmup passes...")
    
    # Warmup passes (not timed)
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
    
    print(f"Running {n_passes} timed passes...")
    times_ms: List[float] = []
    
    for i in range(n_passes):
        # Synchronize before starting timer (for GPU)
        if device == "cuda":
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Synchronize after generation (for GPU)
        if device == "cuda":
            torch.cuda.synchronize()
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed_ms)
        
        # Progress indicator
        if (i + 1) % 10 == 0 or i == n_passes - 1:
            print(f"  Pass {i + 1}/{n_passes}: {elapsed_ms:.2f} ms")
    
    return times_ms


def format_duration(ms: float) -> str:
    """Format milliseconds as human-readable duration."""
    if ms < 1000:
        return f"{ms:.1f} ms"
    elif ms < 60_000:
        return f"{ms / 1000:.1f} seconds"
    elif ms < 3_600_000:
        return f"{ms / 60_000:.1f} minutes"
    else:
        return f"{ms / 3_600_000:.2f} hours"


def main():
    parser = argparse.ArgumentParser(
        description="Measure inference time for PRP comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-xl",
        help="HuggingFace model identifier (default: google/flan-t5-xl)"
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=512,
        help="Target number of prompt tokens to simulate (default: 512)"
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=100,
        help="Number of forward passes to average (default: 100)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup passes excluded from timing (default: 5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="8bit",
        choices=["8bit", "4bit", "none"],
        help="Quantization level (default: 8bit)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
        help="Max tokens to generate (default: 4, same as oracle)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for model files"
    )
    
    args = parser.parse_args()
    
    # Handle 'none' quantization
    quantization = args.quantization if args.quantization != "none" else None
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Print header
    print("=" * 60)
    print("INFERENCE TIME EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}", end="")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"Quantization: {args.quantization}")
    print(f"Target prompt tokens: {args.prompt_tokens}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Passes: {args.num_passes} (+ {args.warmup} warmup)")
    print("=" * 60)
    print()
    
    # Load model
    model, tokenizer = load_model(
        args.model,
        args.device,
        quantization,
        args.cache_dir
    )
    
    # Generate dummy prompt
    print("Generating dummy prompt...")
    prompt = generate_dummy_prompt(tokenizer, args.prompt_tokens)
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
    print(f"Generated prompt with {actual_tokens} tokens\n")
    
    # Run timing experiment
    times_ms = measure_inference_time(
        model,
        tokenizer,
        prompt,
        args.device,
        args.max_new_tokens,
        args.num_passes,
        args.warmup
    )
    
    # Calculate statistics
    mean_time = statistics.mean(times_ms)
    std_time = statistics.stdev(times_ms) if len(times_ms) > 1 else 0
    min_time = min(times_ms)
    max_time = max(times_ms)
    
    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Mean inference time: {mean_time:.2f} ms")
    print(f"Std deviation:       {std_time:.2f} ms")
    print(f"Min:                 {min_time:.2f} ms")
    print(f"Max:                 {max_time:.2f} ms")
    print()
    
    # Projections
    print("PROJECTIONS")
    print("-" * 40)
    for n_comparisons in [100, 1_000, 10_000, 100_000]:
        total_ms = mean_time * n_comparisons
        print(f"{n_comparisons:>10,} comparisons: ~{format_duration(total_ms)}")
    
    print()
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
