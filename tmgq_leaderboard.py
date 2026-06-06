import argparse
import csv
from pathlib import Path


def as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser(description="Print a compact leaderboard from a TMG-Q sweep CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--sort", choices=("ppl", "ratio", "delta"), default="ppl")
    args = parser.parse_args()

    path = Path(args.csv_path)
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    quantized = [
        row
        for row in rows
        if row.get("kind") == "quantized"
        and (row.get("status") or "completed") == "completed"
        and as_float(row.get("ppl")) is not None
        and as_float(row.get("matched_ratio")) is not None
    ]
    baseline = next((row for row in rows if row.get("kind") == "baseline"), None)

    if baseline:
        print(f"Baseline PPL: {baseline.get('ppl')}")
        print()

    sort_key = {
        "ppl": lambda row: as_float(row.get("ppl"), float("inf")),
        "delta": lambda row: as_float(row.get("ppl_delta"), float("inf")),
        "ratio": lambda row: -as_float(row.get("matched_ratio"), 0.0),
    }[args.sort]

    print("rank,bits,group,outlier,svd,quantizer,mixed,calibrated,ppl,delta,ratio,checkpoint")
    for idx, row in enumerate(sorted(quantized, key=sort_key), 1):
        print(
            f"{idx},"
            f"{row.get('bits')},"
            f"{row.get('group_size')},"
            f"{row.get('outlier_fraction')},"
            f"{row.get('svd_rank')},"
            f"{row.get('quantizer')},"
            f"{row.get('mixed_policy')},"
            f"{row.get('calibrate')},"
            f"{row.get('ppl')},"
            f"{row.get('ppl_delta')},"
            f"{row.get('matched_ratio')},"
            f"{row.get('checkpoint')}"
        )


if __name__ == "__main__":
    main()
