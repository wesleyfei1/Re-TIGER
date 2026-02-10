import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from models.rqvae import RQVAE


def kmeans_torch(
    x: torch.Tensor,
    k: int,
    iters: int = 20,
    seed: int = 11,
    batch_size: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator(device=x.device)
    rng.manual_seed(seed)
    indices = torch.randperm(x.shape[0], generator=rng, device=x.device)[:k]
    centers = x[indices].clone()

    for _ in range(iters):
        counts = torch.zeros(k, device=x.device)
        sums = torch.zeros_like(centers)
        assignments = []

        for i in range(0, x.shape[0], batch_size):
            batch = x[i : i + batch_size]
            dists = torch.cdist(batch, centers, p=2)
            batch_idx = torch.argmin(dists, dim=1)
            assignments.append(batch_idx)
            counts += torch.bincount(batch_idx, minlength=k).float()
            sums.index_add_(0, batch_idx, batch)

        assignments = torch.cat(assignments, dim=0)
        mask = counts > 0
        centers[mask] = sums[mask] / counts[mask].unsqueeze(1)

    return centers, assignments


def residual_kmeans_init(
    x: torch.Tensor,
    codebook_size: int,
    num_codebooks: int,
    iters: int = 20,
    seed: int = 11,
    batch_size: int = 8192,
) -> torch.Tensor:
    residual = x
    codebooks = []

    for i in range(num_codebooks):
        centers, assignments = kmeans_torch(
            residual,
            k=codebook_size,
            iters=iters,
            seed=seed + i,
            batch_size=batch_size,
        )
        codebooks.append(centers)
        residual = residual - centers[assignments]

    return torch.stack(codebooks, dim=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize RQ-VAE codebooks with k-means.")
    parser.add_argument(
        "--embeddings",
        default="~/datasets/processed/beauty/sentence_t5_embeddings.npy",
        help="Path to sentence_t5_embeddings.npy",
    )
    parser.add_argument(
        "--output",
        default="~/datasets/processed/beauty/rqvae_codebooks.pt",
        help="Path to save codebooks",
    )
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--num_codebooks", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument(
        "--hidden_dims",
        default="512,256,128",
        help="Comma-separated encoder hidden dims",
    )
    parser.add_argument(
        "--encoder_ckpt",
        default="",
        help="Optional RQ-VAE checkpoint for encoder initialization",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def _parse_hidden_dims(value: str) -> List[int]:
    if not value:
        return []
    return [int(x) for x in value.split(",") if x]


def main():
    args = parse_args()
    embeddings_path = Path(args.embeddings).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not embeddings_path.exists():
        raise SystemExit(f"Embeddings file not found: {embeddings_path}")

    embeddings = np.load(embeddings_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(embeddings).float().to(device)

    use_encoder = args.encoder_ckpt or args.latent_dim != x.shape[1]
    if use_encoder:
        hidden_dims = _parse_hidden_dims(args.hidden_dims)
        model = RQVAE(
            input_dim=x.shape[1],
            hidden_dims=hidden_dims,
            latent_dim=args.latent_dim,
            codebook_size=args.codebook_size,
            num_codebooks=args.num_codebooks,
        ).to(device)
        if args.encoder_ckpt:
            payload = torch.load(Path(args.encoder_ckpt).expanduser(), map_location=device)
            model.load_state_dict(payload.get("model", payload), strict=False)
        with torch.no_grad():
            x_for_kmeans = model.encoder(x)
    else:
        x_for_kmeans = x

    codebooks = residual_kmeans_init(
        x_for_kmeans,
        codebook_size=args.codebook_size,
        num_codebooks=args.num_codebooks,
        iters=args.iters,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    torch.save(
        {
            "codebooks": codebooks.cpu(),
            "codebook_size": args.codebook_size,
            "num_codebooks": args.num_codebooks,
            "latent_dim": args.latent_dim if use_encoder else x.shape[1],
        },
        output_path,
    )
    print("Saved codebooks:", output_path)


if __name__ == "__main__":
    main()
