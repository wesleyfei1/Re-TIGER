import sys
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from init.codebook_init import residual_kmeans_init
from models.rqvae import RQVAE


def main():
    torch.manual_seed(11)
    np.random.seed(11)

    n = 256
    dim = 768
    centers = torch.randn(4, dim) * 2.0
    labels = torch.randint(0, centers.shape[0], (n,))
    data = centers[labels] + 0.1 * torch.randn(n, dim)

    model = RQVAE(
        input_dim=dim,
        hidden_dims=[128, 64, 32],
        latent_dim=16,
        codebook_size=16,
        num_codebooks=3,
        commit_cost=0.25,
    )
    with torch.no_grad():
        z_e = model.encoder(data)
    codebooks = residual_kmeans_init(
        z_e,
        codebook_size=16,
        num_codebooks=3,
        iters=5,
        seed=11,
        batch_size=128,
    )
    assert codebooks.shape == (3, 16, 16)
    model.quantizer.set_codebooks(codebooks)

    recon, codes, q_loss = model(data)
    assert recon.shape == data.shape
    assert codes.shape == (n, 3)
    assert torch.isfinite(q_loss).all()

    recon_loss = torch.mean((recon - data) ** 2).item()
    print("recon_loss=", recon_loss)
    print("q_loss=", q_loss.item())

    print("OK")


if __name__ == "__main__":
    main()
