import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


PPL_RE = re.compile(r"WikiText-2 PPL:\s*([0-9.]+)")
RATIO_RE = re.compile(r"matched ratio:\s*([0-9.]+)x")
PACKED_RE = re.compile(r"matched packed payload:\s*([0-9.]+)\s*KiB")
FP16_RE = re.compile(r"matched FP16 payload:\s*([0-9.]+)\s*KiB")


def run_cmd(args, timeout=None):
    start = time.time()
    proc = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return {
        "returncode": proc.returncode,
        "seconds": time.time() - start,
        "output": proc.stdout,
    }


def parse_first(pattern, text):
    match = pattern.search(text)
    return float(match.group(1)) if match else None


def safe_name(model_name):
    return model_name.replace("/", "__").replace("\\", "__").replace(":", "_")


def eval_ppl(py, model, checkpoint, max_length, samples, timeout):
    cmd = [py, "tmgq_eval_ppl.py", "--model", model, "--max-length", str(max_length), "--samples", str(samples)]
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    result = run_cmd(cmd, timeout=timeout)
    result["ppl"] = parse_first(PPL_RE, result["output"])
    return result


def export_model(
    py,
    model,
    output,
    bits,
    group_size,
    outlier_fraction,
    calibrate,
    calib_samples,
    calib_length,
    svd_rank,
    quantizer,
    mixed_policy,
    adaptive_2bit_nmse,
    adaptive_3bit_nmse,
    adaptive_probe_rows,
    quant_device,
    target_ratio,
    timeout,
):
    cmd = [
        py,
        "tmgq_export_llama.py",
        "--model",
        model,
        "--bits",
        str(bits),
        "--group-size",
        str(group_size),
        "--output",
        output,
        "--quant-device",
        quant_device,
    ]
    if outlier_fraction > 0:
        cmd.extend(["--outlier-fraction", str(outlier_fraction)])
    if svd_rank > 0:
        cmd.extend(["--svd-rank", str(svd_rank)])
    if quantizer != "linear":
        cmd.extend(["--quantizer", quantizer])
    if mixed_policy != "none":
        cmd.extend(["--mixed-policy", mixed_policy])
    if mixed_policy == "adaptive":
        cmd.extend(
            [
                "--adaptive-2bit-nmse",
                str(adaptive_2bit_nmse),
                "--adaptive-3bit-nmse",
                str(adaptive_3bit_nmse),
                "--adaptive-probe-rows",
                str(adaptive_probe_rows),
            ]
        )
    if mixed_policy == "budget":
        cmd.extend(["--target-ratio", str(target_ratio), "--adaptive-probe-rows", str(adaptive_probe_rows)])
    if calibrate:
        cmd.extend(["--calibrate", "--calib-samples", str(calib_samples), "--calib-length", str(calib_length)])
    return run_cmd(cmd, timeout=timeout)


def report_size(py, model, checkpoint, timeout):
    result = run_cmd([py, "tmgq_checkpoint_report.py", "--model", model, checkpoint], timeout=timeout)
    result["ratio"] = parse_first(RATIO_RE, result["output"])
    result["packed_kib"] = parse_first(PACKED_RE, result["output"])
    result["fp16_kib"] = parse_first(FP16_RE, result["output"])
    return result


def load_existing(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, rows):
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def save_csv(path, rows):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys() if key != "logs"})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})

def is_completed(row):
    return (
        row.get("kind") == "quantized"
        and row.get("status", "completed") == "completed"
        and row.get("ppl") is not None
        and row.get("matched_ratio") is not None
        and row.get("returncode_export", 0) == 0
        and row.get("returncode_eval", 0) == 0
        and row.get("returncode_size", 0) == 0
    )


def main():
    parser = argparse.ArgumentParser(description="Run TMG-Q export/eval sweeps.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--calib-samples", type=int, default=16)
    parser.add_argument("--calib-length", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--adaptive-2bit-nmse", type=float, default=0.06)
    parser.add_argument("--adaptive-3bit-nmse", type=float, default=0.02)
    parser.add_argument("--adaptive-probe-rows", type=int, default=256)
    parser.add_argument("--quant-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--target-ratio", type=float, default=3.5)
    parser.add_argument(
        "--max-ppl-increase",
        type=float,
        default=None,
        help="Reject a result when PPL rises by more than this percentage over baseline.",
    )
    parser.add_argument("--configs", default="4:128:0:false:0:linear:none,3:128:0.001:false:0:linear:none,3:128:0.01:false:0:linear:none")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tag = args.tag or safe_name(args.model)
    out_dir = Path("sweep_results") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"
    rows = load_existing(json_path)

    if not args.skip_baseline and not any(row.get("kind") == "baseline" for row in rows):
        print("Running baseline PPL...")
        eval_result = eval_ppl(args.python, args.model, None, args.max_length, args.samples, args.timeout)
        rows.append({
            "kind": "baseline",
            "model": args.model,
            "ppl": eval_result["ppl"],
            "seconds_eval": round(eval_result["seconds"], 3),
            "returncode": eval_result["returncode"],
            "logs": {"eval": eval_result["output"]},
        })
        save_json(json_path, rows)
        save_csv(csv_path, rows)
        print(f"Baseline PPL: {eval_result['ppl']}")

    for raw_cfg in [part.strip() for part in args.configs.split(",") if part.strip()]:
        parts = raw_cfg.split(":")
        if len(parts) == 4:
            bits_s, group_s, outlier_s, calibrate_s = parts
            svd_s = "0"
            quantizer = "linear"
            mixed_policy = "none"
        elif len(parts) == 5:
            bits_s, group_s, outlier_s, calibrate_s, svd_s = parts
            quantizer = "linear"
            mixed_policy = "none"
        elif len(parts) == 6:
            bits_s, group_s, outlier_s, calibrate_s, svd_s, quantizer = parts
            mixed_policy = "none"
        elif len(parts) == 7:
            bits_s, group_s, outlier_s, calibrate_s, svd_s, quantizer, mixed_policy = parts
        else:
            raise ValueError(
                f"Bad config '{raw_cfg}'. "
                "Expected bits:group:outlier:calibrate[:svd_rank[:quantizer[:mixed_policy]]]"
            )
        bits = int(bits_s)
        group_size = int(group_s)
        outlier_fraction = float(outlier_s)
        calibrate = calibrate_s.lower() in ("1", "true", "yes", "y")
        svd_rank = int(svd_s)
        ckpt = out_dir / (
            f"{tag}_{bits}bit_g{group_size}_o{outlier_fraction:g}_svd{svd_rank}_"
            f"{quantizer}_{mixed_policy}_{'cal' if calibrate else 'raw'}.pt"
        )

        if any(row.get("checkpoint") == str(ckpt) and is_completed(row) for row in rows) and not args.force:
            print(f"Skipping existing result: {ckpt}")
            continue

        if not ckpt.exists() or args.force:
            print(f"Exporting {ckpt.name}...")
            export_result = export_model(
                args.python,
                args.model,
                str(ckpt),
                bits,
                group_size,
                outlier_fraction,
                calibrate,
                args.calib_samples,
                args.calib_length,
                svd_rank,
                quantizer,
                mixed_policy,
                args.adaptive_2bit_nmse,
                args.adaptive_3bit_nmse,
                args.adaptive_probe_rows,
                args.quant_device,
                args.target_ratio,
                args.timeout,
            )
        else:
            export_result = {"returncode": 0, "seconds": 0.0, "output": "checkpoint already existed"}

        if export_result["returncode"] != 0 or not ckpt.exists():
            print(f"Export failed for {ckpt.name}; skipping evaluation.")
            eval_result = {"returncode": None, "seconds": 0.0, "output": "", "ppl": None}
            size_result = {
                "returncode": None,
                "seconds": 0.0,
                "output": "",
                "ratio": None,
                "packed_kib": None,
                "fp16_kib": None,
            }
            status = "export_failed"
        else:
            print(f"Evaluating {ckpt.name}...")
            eval_result = eval_ppl(args.python, args.model, str(ckpt), args.max_length, args.samples, args.timeout)
            if eval_result["returncode"] == 0 and eval_result["ppl"] is not None:
                size_result = report_size(args.python, args.model, str(ckpt), args.timeout)
                status = "completed" if size_result["returncode"] == 0 and size_result["ratio"] is not None else "size_failed"
            else:
                size_result = {
                    "returncode": None,
                    "seconds": 0.0,
                    "output": "",
                    "ratio": None,
                    "packed_kib": None,
                    "fp16_kib": None,
                }
                status = "eval_failed"
        baseline = next((row for row in rows if row.get("kind") == "baseline"), {})
        base_ppl = baseline.get("ppl")
        ppl_delta = eval_result["ppl"] - base_ppl if eval_result["ppl"] is not None and base_ppl is not None else None
        ppl_increase_percent = (
            100.0 * ppl_delta / base_ppl
            if ppl_delta is not None and base_ppl is not None and base_ppl > 0
            else None
        )
        if (
            status == "completed"
            and args.max_ppl_increase is not None
            and ppl_increase_percent is not None
            and ppl_increase_percent > args.max_ppl_increase
        ):
            status = "quality_rejected"

        row = {
            "kind": "quantized",
            "status": status,
            "model": args.model,
            "checkpoint": str(ckpt),
            "bits": bits,
            "group_size": group_size,
            "outlier_fraction": outlier_fraction,
            "svd_rank": svd_rank,
            "quantizer": quantizer,
            "mixed_policy": mixed_policy,
            "adaptive_2bit_nmse": args.adaptive_2bit_nmse if mixed_policy == "adaptive" else None,
            "adaptive_3bit_nmse": args.adaptive_3bit_nmse if mixed_policy == "adaptive" else None,
            "adaptive_probe_rows": args.adaptive_probe_rows if mixed_policy == "adaptive" else None,
            "quant_device": args.quant_device,
            "target_ratio": args.target_ratio if mixed_policy == "budget" else None,
            "calibrate": calibrate,
            "ppl": eval_result["ppl"],
            "ppl_delta": ppl_delta,
            "ppl_increase_percent": ppl_increase_percent,
            "max_ppl_increase": args.max_ppl_increase,
            "matched_ratio": size_result["ratio"],
            "matched_packed_kib": size_result["packed_kib"],
            "matched_fp16_kib": size_result["fp16_kib"],
            "checkpoint_mb": ckpt.stat().st_size / (1024 * 1024) if ckpt.exists() else None,
            "seconds_export": round(export_result["seconds"], 3),
            "seconds_eval": round(eval_result["seconds"], 3),
            "returncode_export": export_result["returncode"],
            "returncode_eval": eval_result["returncode"],
            "returncode_size": size_result["returncode"],
            "logs": {
                "export": export_result["output"],
                "eval": eval_result["output"],
                "size": size_result["output"],
            },
        }
        rows = [existing for existing in rows if existing.get("checkpoint") != str(ckpt)]
        rows.append(row)
        save_json(json_path, rows)
        save_csv(csv_path, rows)
        print(f"Result: bits={bits} group={group_size} outlier={outlier_fraction} cal={calibrate} PPL={row['ppl']} ratio={row['matched_ratio']}")

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
