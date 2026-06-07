import argparse
import math

import torch
from tmgq_transformers_compat import disable_broken_torchvision

disable_broken_torchvision()
from transformers import AutoModelForCausalLM, AutoTokenizer

from tmgq_friend_chat import load_packed_huggingface

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def get_chunks(tokenizer, split="test", max_length=256, max_samples=128):
    if load_dataset is None:
        raise RuntimeError("Evaluation requires the 'datasets' package. Install it with: pip install datasets")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in dataset["text"] if t.strip())
    tokenized = tokenizer(text, return_tensors="pt")["input_ids"][0]

    chunks = []
    for start in range(0, len(tokenized) - max_length, max_length):
        if len(chunks) >= max_samples:
            break
        chunks.append(tokenized[start:start + max_length].unsqueeze(0))
    return chunks


@torch.no_grad()
def evaluate_ppl(model, tokenizer, device, max_length=256, max_samples=128):
    chunks = get_chunks(tokenizer, max_length=max_length, max_samples=max_samples)
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for idx, chunk in enumerate(chunks, 1):
        print(f"  Eval block {idx}/{len(chunks)}", end="\r")
        chunk = chunk.to(device)
        outputs = model(chunk, labels=chunk)
        loss = outputs.loss.item()
        if math.isfinite(loss):
            total_loss += loss * chunk.shape[1]
            total_tokens += chunk.shape[1]

    print()
    return math.exp(total_loss / max(1, total_tokens))


def load_original(model_name, dtype):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(device)
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser(description="Evaluate WikiText-2 perplexity for original or TMG-Q packed models.")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--checkpoint", default=None, help="Packed TMG-Q .pt checkpoint. Omit to evaluate the original model.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    args = parser.parse_args()

    if args.checkpoint:
        model, tokenizer, device = load_packed_huggingface(args.model, args.checkpoint)
        label = args.checkpoint
    else:
        model, tokenizer, device = load_original(args.model, args.dtype)
        label = f"{args.model} original"

    ppl = evaluate_ppl(model, tokenizer, device, max_length=args.max_length, max_samples=args.samples)
    print(f"{label} WikiText-2 PPL: {ppl:.4f}")


if __name__ == "__main__":
    main()
