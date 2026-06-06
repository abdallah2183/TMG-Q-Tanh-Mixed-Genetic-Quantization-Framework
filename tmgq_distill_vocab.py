import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from tmgq_transformers_compat import disable_broken_torchvision

disable_broken_torchvision()
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tmgq_friend_chat import load_packed_huggingface
from tmgq_packer import (
    DistilledQuantizedEmbedding,
    DistilledTiedLMHead,
    QuantizedEmbedding,
    QuantizedLinear,
)


def calibration_chunks(tokenizer, sequence_length, samples):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(item for item in dataset["text"] if item.strip())
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    return [
        tokens[start:start + sequence_length].unsqueeze(0)
        for start in range(0, min(len(tokens) - sequence_length, samples * sequence_length), sequence_length)
    ]


def bake_residual(model, distilled_embedding):
    frozen = distilled_embedding.frozen_embedding
    frozen.svd_u = distilled_embedding.residual_left.detach().to(torch.float16).cpu()
    frozen.svd_s = torch.ones(
        distilled_embedding.residual_left.shape[1],
        dtype=torch.float16,
    )
    frozen.svd_v = distilled_embedding.residual_right.detach().to(torch.float16).cpu()
    frozen.tied_lm_head = torch.tensor(True, dtype=torch.bool)
    model.set_input_embeddings(frozen)
    from tmgq_packer import QuantizedTiedLMHead
    model.set_output_embeddings(QuantizedTiedLMHead(frozen))

@torch.no_grad()
def validation_kl(teacher, student, chunks, temperature):
    total = 0.0
    for input_ids in chunks:
        input_ids = input_ids.to("cuda")
        teacher_logits = teacher(input_ids).logits.float()
        student_logits = student(input_ids).logits.float()
        loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2) / input_ids.shape[1]
        total += loss.item()
    return total / max(1, len(chunks))

def enable_quantized_layer_tuning(model, maximum_bits):
    trainable = []
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, QuantizedLinear):
            continue
        bits = int(module.n_bits.item())
        if bits > maximum_bits:
            continue
        if module.codebooks.numel() > 0:
            value = module.codebooks.detach().float()
            del module._buffers["codebooks"]
            module.register_parameter("codebooks", torch.nn.Parameter(value))
            trainable.append(module.codebooks)
            names.append(f"{name}.codebooks")
        elif module.scales.numel() > 0:
            scales = module.scales.detach().float()
            zeros = module.zeros.detach().float()
            del module._buffers["scales"]
            del module._buffers["zeros"]
            module.register_parameter("scales", torch.nn.Parameter(scales))
            module.register_parameter("zeros", torch.nn.Parameter(zeros))
            trainable.extend((module.scales, module.zeros))
            names.extend((f"{name}.scales", f"{name}.zeros"))
    return trainable, names

def bake_quantized_layer_parameters(model):
    for module in model.modules():
        if not isinstance(module, QuantizedLinear):
            continue
        for name in ("codebooks", "scales", "zeros"):
            value = getattr(module, name)
            if not isinstance(value, torch.nn.Parameter):
                continue
            del module._parameters[name]
            module.register_buffer(name, value.detach().to(torch.float16))


def distill(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Vocabulary distillation requires CUDA")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    student, _, _ = load_packed_huggingface(args.model, args.checkpoint)
    frozen_embedding = student.get_input_embeddings()
    if not isinstance(frozen_embedding, QuantizedEmbedding):
        raise RuntimeError("Checkpoint must contain a QuantizedEmbedding")

    for parameter in student.parameters():
        parameter.requires_grad_(False)
    distilled = DistilledQuantizedEmbedding(frozen_embedding, args.rank).to(device)
    if frozen_embedding.svd_s.numel() == args.rank:
        with torch.no_grad():
            distilled.residual_left.copy_(
                frozen_embedding.svd_u.float()
                * frozen_embedding.svd_s.float().unsqueeze(0)
            )
            distilled.residual_right.copy_(frozen_embedding.svd_v.float())
        frozen_embedding.svd_u = torch.empty(0, dtype=torch.float16, device=device)
        frozen_embedding.svd_s = torch.empty(0, dtype=torch.float16, device=device)
        frozen_embedding.svd_v = torch.empty(0, dtype=torch.float16, device=device)
        print(f"Continuing from checkpoint rank-{args.rank} residual.")
    elif args.initialize_svd:
        with torch.no_grad():
            original = teacher.get_input_embeddings().weight.float()
            quantized = frozen_embedding.dequantize(torch.float32)
            distilled.initialize_from_residual(original - quantized)
    student.set_input_embeddings(distilled)
    student.set_output_embeddings(DistilledTiedLMHead(distilled))
    student.eval()
    quant_parameters = []
    if args.tune_quant_layers:
        quant_parameters, quant_names = enable_quantized_layer_tuning(
            student,
            args.tune_max_bits,
        )
        print(f"Tuning {len(quant_names)} packed tensors at <= {args.tune_max_bits}-bit.")

    all_chunks = calibration_chunks(
        tokenizer,
        args.sequence_length,
        args.samples + args.validation_samples,
    )
    chunks = all_chunks[:args.samples]
    validation_chunks = all_chunks[args.samples:]
    parameter_groups = [
        {
            "params": [distilled.residual_left, distilled.residual_right],
            "lr": args.learning_rate,
        }
    ]
    if quant_parameters:
        parameter_groups.append(
            {
                "params": quant_parameters,
                "lr": args.quant_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    best_loss = math.inf
    best_validation = validation_kl(
        teacher,
        student,
        validation_chunks,
        args.temperature,
    )
    best_left = distilled.residual_left.detach().cpu().clone()
    best_right = distilled.residual_right.detach().cpu().clone()
    best_quant = [parameter.detach().cpu().clone() for parameter in quant_parameters]
    print(f"Initial validation KL: {best_validation:.6f}")

    for step in range(args.steps):
        input_ids = chunks[step % len(chunks)].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(input_ids).logits.float()
        with torch.autocast("cuda", dtype=torch.float16 if args.fp16 else torch.bfloat16):
            student_logits = student(input_ids).logits.float()
            temperature = args.temperature
            kl = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2) / input_ids.shape[1]
            ce = F.cross_entropy(
                student_logits[:, :-1].reshape(-1, student_logits.shape[-1]),
                input_ids[:, 1:].reshape(-1),
            )
            loss = args.kl_weight * kl + (1.0 - args.kl_weight) * ce
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [distilled.residual_left, distilled.residual_right, *quant_parameters],
            args.max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        best_loss = min(best_loss, loss.item())
        if step == 0 or (step + 1) % args.log_every == 0:
            val_kl = validation_kl(
                teacher,
                student,
                validation_chunks,
                args.temperature,
            )
            if val_kl < best_validation:
                best_validation = val_kl
                best_left = distilled.residual_left.detach().cpu().clone()
                best_right = distilled.residual_right.detach().cpu().clone()
                best_quant = [
                    parameter.detach().cpu().clone()
                    for parameter in quant_parameters
                ]
            print(
                f"step {step + 1}/{args.steps} loss={loss.item():.6f} "
                f"kl={kl.item():.6f} ce={ce.item():.6f} val_kl={val_kl:.6f}"
            )

    distilled.residual_left.data.copy_(best_left.to(device))
    distilled.residual_right.data.copy_(best_right.to(device))
    for parameter, best_value in zip(quant_parameters, best_quant):
        parameter.data.copy_(best_value.to(device))
    bake_residual(student, distilled)
    bake_quantized_layer_parameters(student)
    student = student.cpu()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), args.output)
    print(f"Saved distilled checkpoint: {args.output}")
    print(f"Best training loss: {best_loss:.6f}")
    print(f"Best validation KL: {best_validation:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Distill a compressed tied vocabulary from its FP teacher.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl-weight", type=float, default=0.9)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--initialize-svd", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tune-quant-layers", action="store_true")
    parser.add_argument("--tune-max-bits", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--quant-learning-rate", type=float, default=1e-4)
    distill(parser.parse_args())


if __name__ == "__main__":
    main()
