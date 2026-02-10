from typing import Tuple

import torch
from torch import nn


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embedding = nn.Embedding(4096, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _add_pos(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return self.embedding(x) + self.pos_embedding(positions)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src_emb = self._add_pos(src)
        tgt_emb = self._add_pos(tgt)
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)
        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.lm_head(out)

    def greedy_decode(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = src.size(0)
        decoded = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            logits = self.forward(
                src,
                decoded,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=decoded.eq(self.pad_id),
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            decoded = torch.cat([decoded, next_token], dim=1)
            if torch.all(next_token.squeeze(1) == eos_id):
                break
        return decoded

    @staticmethod
    def _generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((size, size), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)
