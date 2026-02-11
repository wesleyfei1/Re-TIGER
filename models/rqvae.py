import math
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ResidualQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        num_codebooks: int,
        commit_cost: float = 0.25,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        dead_code_threshold: int = 10,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.commit_cost = commit_cost
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.dead_code_threshold = dead_code_threshold
        self.codebooks = nn.Parameter(
            torch.randn(num_codebooks, codebook_size, dim) / math.sqrt(dim)
        )
        self.register_buffer(
            "ema_cluster_size", torch.zeros(num_codebooks, codebook_size)
        )
        self.register_buffer(
            "ema_codebook", torch.zeros(num_codebooks, codebook_size, dim)
        )

    def _nearest_code(
        self, x: torch.Tensor, codebook: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flat = x.view(-1, self.dim)
        x_norm = (x_flat ** 2).sum(dim=1, keepdim=True)
        c_norm = (codebook ** 2).sum(dim=1).unsqueeze(0)
        distances = x_norm - 2 * x_flat @ codebook.t() + c_norm
        indices = torch.argmin(distances, dim=1)
        quantized = codebook[indices].view_as(x)
        return quantized, indices.view(x.shape[0], -1)

    def _ema_update(self, codebook_idx: int, x: torch.Tensor, indices: torch.Tensor) -> None:
        flat_idx = indices.view(-1)
        one_hot = F.one_hot(flat_idx, num_classes=self.codebook_size).float()
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.t() @ x.view(-1, self.dim)

        self.ema_cluster_size[codebook_idx] = (
            self.ema_cluster_size[codebook_idx] * self.ema_decay
            + (1 - self.ema_decay) * cluster_size
        )
        self.ema_codebook[codebook_idx] = (
            self.ema_codebook[codebook_idx] * self.ema_decay
            + (1 - self.ema_decay) * embed_sum
        )
        n = self.ema_cluster_size[codebook_idx] + self.ema_eps
        self.codebooks.data[codebook_idx] = self.ema_codebook[codebook_idx] / n.unsqueeze(1)

        if self.dead_code_threshold > 0:
            dead = self.ema_cluster_size[codebook_idx] < self.dead_code_threshold
            if dead.any():
                rand_idx = torch.randint(
                    0,
                    x.view(-1, self.dim).size(0),
                    (dead.sum().item(),),
                    device=x.device,
                )
                self.codebooks.data[codebook_idx, dead] = x.view(-1, self.dim)[rand_idx]

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z_e
        quantized_sum = torch.zeros_like(z_e)
        all_codes = []

        for i in range(self.num_codebooks):
            codebook = self.codebooks[i]
            quantized, indices = self._nearest_code(residual, codebook)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            all_codes.append(indices.squeeze(1))
            if self.training:
                self._ema_update(i, residual.detach() + quantized.detach(), indices)

        codes = torch.stack(all_codes, dim=1)
        codebook_loss = F.mse_loss(quantized_sum.detach(), z_e)
        commit_loss = F.mse_loss(quantized_sum, z_e.detach())
        loss = codebook_loss + self.commit_cost * commit_loss
        z_q = z_e + (quantized_sum - z_e).detach()
        return z_q, codes, loss

    def set_codebooks(self, codebooks: torch.Tensor) -> None:
        if codebooks.shape != self.codebooks.shape:
            raise ValueError(
                f"Codebook shape mismatch: expected {self.codebooks.shape}, got {codebooks.shape}"
            )
        with torch.no_grad():
            self.codebooks.copy_(codebooks)


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: List[int] | None = None,
        latent_dim: int = 32,
        codebook_size: int = 256,
        num_codebooks: int = 3,
        commit_cost: float = 0.25,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        dead_code_threshold: int = 10,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.encoder = self._build_mlp([input_dim] + hidden_dims + [latent_dim])
        self.decoder = self._build_mlp([latent_dim] + list(reversed(hidden_dims)) + [input_dim])
        self.quantizer = ResidualQuantizer(
            dim=latent_dim,
            codebook_size=codebook_size,
            num_codebooks=num_codebooks,
            commit_cost=commit_cost,
            ema_decay=ema_decay,
            ema_eps=ema_eps,
            dead_code_threshold=dead_code_threshold,
        )

    @staticmethod
    def _build_mlp(dims: List[int]) -> nn.Sequential:
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encode(x)
        z_q, codes, q_loss = self.quantize(z_e)
        recon = self.decode(z_q)
        return recon, codes, q_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.quantizer(z_e)

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        codes_list = []
        for i in range(0, x.shape[0], batch_size):
            batch = x[i : i + batch_size]
            z_e = self.encode(batch)
            _, codes, _ = self.quantize(z_e)
            codes_list.append(codes.cpu())
        return torch.cat(codes_list, dim=0)
