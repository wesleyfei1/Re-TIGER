from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def compute_usage(codes: np.ndarray, codebook_size: int) -> Tuple[Dict[str, float], np.ndarray]:
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2D (N, M), got shape {codes.shape}")
    max_code = int(codes.max()) if codes.size else -1
    if max_code >= codebook_size:
        raise ValueError(
            f"codebook_size {codebook_size} is too small for max code {max_code}."
        )
    num_codebooks = codes.shape[1]
    usage = np.zeros((num_codebooks, codebook_size), dtype=np.int64)
    for i in range(num_codebooks):
        counts = np.bincount(codes[:, i], minlength=codebook_size)
        usage[i] = counts
    stats = {
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
        "coverage_per_codebook": float(np.mean((usage > 0).sum(axis=1) / codebook_size)),
        "coverage_min": float(((usage > 0).sum(axis=1) / codebook_size).min()),
        "coverage_max": float(((usage > 0).sum(axis=1) / codebook_size).max()),
    }
    return stats, usage


def parse_args():
    parser = argparse.ArgumentParser(description="Measure codebook usage from semantic ids.")
    parser.add_argument(
        "--semantic_ids",
        default="~/datasets/processed/beauty/semantic_ids.json",
        help="Path to semantic_ids.json",
    )
    parser.add_argument("--codebook_size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    semantic_path = Path(args.semantic_ids).expanduser()
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    codes = np.array(list(semantic.values()), dtype=np.int64)

    stats, usage = compute_usage(codes, args.codebook_size)
    print("Codebook usage stats:")
    for key, val in stats.items():
        print(f"{key}: {val}")
    print("Per-codebook coverage:")
    coverage = (usage > 0).sum(axis=1) / args.codebook_size
    print(coverage.tolist())


if __name__ == "__main__":
    main()
