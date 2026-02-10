import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, Dataset

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from models.seq_to_seq import Seq2SeqTransformer


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
CODE_OFFSET = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Train seq-to-seq retrieval model.")
    parser.add_argument("--train", default="~/datasets/processed/beauty/train.jsonl")
    parser.add_argument("--valid", default="~/datasets/processed/beauty/valid.jsonl")
    parser.add_argument("--semantic_ids", default="~/datasets/processed/beauty/semantic_ids.json")
    parser.add_argument("--output_dir", default="~/datasets/processed/beauty")
    parser.add_argument("--max_src_len", type=int, default=128)
    parser.add_argument("--max_tgt_len", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--user_vocab", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb_project", default="retiger")
    parser.add_argument("--wandb_run_name", default="")
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_semantic_map(path: Path) -> Dict[str, List[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: [int(x) for x in v] for k, v in raw.items()}


def build_user_sequences(path: Path) -> Dict[str, List[str]]:
    user_events: Dict[str, List[Tuple[int, str]]] = {}
    for rec in read_jsonl(path):
        user_events.setdefault(rec["user_id"], []).append((rec["timestamp"], rec["item_id"]))
    sequences: Dict[str, List[str]] = {}
    for user_id, events in user_events.items():
        events.sort(key=lambda x: x[0])
        sequences[user_id] = [item_id for _, item_id in events]
    return sequences


def hash_user(user_id: str, user_vocab: int) -> int:
    return abs(hash(user_id)) % user_vocab


def tokens_from_item(item_id: str, semantic_map: Dict[str, List[int]]) -> List[int]:
    return [CODE_OFFSET + t for t in semantic_map[item_id]]


def build_examples(
    sequences: Dict[str, List[str]],
    semantic_map: Dict[str, List[int]],
) -> List[Tuple[str, List[int], List[int]]]:
    examples = []
    for user_id, items in sequences.items():
        if len(items) < 2:
            continue
        history = []
        for idx in range(len(items) - 1):
            next_item = items[idx + 1]
            history.extend(tokens_from_item(items[idx], semantic_map))
            target_tokens = tokens_from_item(next_item, semantic_map)
            examples.append((user_id, list(history), target_tokens))
    return examples


class SeqDataset(Dataset):
    def __init__(
        self,
        examples: List[Tuple[str, List[int], List[int]]],
        user_vocab: int,
        code_vocab: int,
        max_src_len: int,
        max_tgt_len: int,
    ) -> None:
        self.examples = examples
        self.user_vocab = user_vocab
        self.code_vocab = code_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_id, history, target = self.examples[idx]
        user_token = CODE_OFFSET + self.code_vocab + hash_user(user_id, self.user_vocab)
        src = [user_token] + history
        src = src[-self.max_src_len :]
        tgt = target[: self.max_tgt_len]
        dec_in = [BOS_ID] + tgt
        dec_out = tgt + [EOS_ID]
        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt_in": torch.tensor(dec_in, dtype=torch.long),
            "tgt_out": torch.tensor(dec_out, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    src_lens = [len(b["src"]) for b in batch]
    tgt_lens = [len(b["tgt_in"]) for b in batch]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_batch = torch.full((len(batch), max_src), PAD_ID, dtype=torch.long)
    tgt_in_batch = torch.full((len(batch), max_tgt), PAD_ID, dtype=torch.long)
    tgt_out_batch = torch.full((len(batch), max_tgt), PAD_ID, dtype=torch.long)

    for i, sample in enumerate(batch):
        src = sample["src"]
        tgt_in = sample["tgt_in"]
        tgt_out = sample["tgt_out"]
        src_batch[i, : src.size(0)] = src
        tgt_in_batch[i, : tgt_in.size(0)] = tgt_in
        tgt_out_batch[i, : tgt_out.size(0)] = tgt_out

    return {
        "src": src_batch,
        "tgt_in": tgt_in_batch,
        "tgt_out": tgt_out_batch,
        "src_pad": src_batch.eq(PAD_ID),
        "tgt_pad": tgt_in_batch.eq(PAD_ID),
    }


def build_vocab_size(semantic_map: Dict[str, List[int]], user_vocab: int) -> int:
    max_token = 0
    for tokens in semantic_map.values():
        if tokens:
            max_token = max(max_token, max(tokens))
    code_vocab = max_token + 1
    return CODE_OFFSET + code_vocab + user_vocab


def inverse_sqrt_schedule(step: int, warmup: int, base_lr: float) -> float:
    if step <= warmup:
        return base_lr
    return base_lr * math.sqrt(warmup / step)


def train_loop(
    model: Seq2SeqTransformer,
    loader: DataLoader,
    args,
    device: torch.device,
    wandb,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    step = 0
    model.train()

    while step < args.steps:
        for batch in loader:
            step += 1
            optimizer.param_groups[0]["lr"] = inverse_sqrt_schedule(step, args.warmup_steps, args.lr)
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            src_pad = batch["src_pad"].to(device)
            tgt_pad = batch["tgt_pad"].to(device)

            optimizer.zero_grad()
            logits = model(src, tgt_in, src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                wandb.log(
                    {"step": step, "loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
                )
                print(f"Step {step}: loss={loss.item():.6f}")

            if step >= args.steps:
                break


def run_inference(
    model: Seq2SeqTransformer,
    semantic_map: Dict[str, List[int]],
    user_id: str,
    history_items: List[str],
    user_vocab: int,
    max_src_len: int,
    max_tgt_len: int,
    device: torch.device,
) -> List[int]:
    max_token = 0
    for tokens in semantic_map.values():
        if tokens:
            max_token = max(max_token, max(tokens))
    code_vocab = max_token + 1
    user_token = CODE_OFFSET + code_vocab + hash_user(user_id, user_vocab)
    history_tokens: List[int] = []
    for item_id in history_items:
        history_tokens.extend(tokens_from_item(item_id, semantic_map))
    src = [user_token] + history_tokens
    src = src[-max_src_len:]
    src_tensor = torch.tensor([src], dtype=torch.long, device=device)
    src_pad = src_tensor.eq(PAD_ID)
    decoded = model.greedy_decode(
        src_tensor,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        max_len=max_tgt_len,
        src_key_padding_mask=src_pad,
    )
    return decoded[0].tolist()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    train_path = Path(args.train).expanduser()
    semantic_path = Path(args.semantic_ids).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    semantic_map = build_semantic_map(semantic_path)
    sequences = build_user_sequences(train_path)
    examples = build_examples(sequences, semantic_map)

    max_token = 0
    for tokens in semantic_map.values():
        if tokens:
            max_token = max(max_token, max(tokens))
    code_vocab = max_token + 1

    dataset = SeqDataset(
        examples,
        args.user_vocab,
        code_vocab,
        args.max_src_len,
        args.max_tgt_len,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    vocab_size = build_vocab_size(semantic_map, args.user_vocab)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        pad_id=PAD_ID,
    ).to(device)


    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or None,
        config={
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "layers": args.layers,
            "dropout": args.dropout,
            "user_vocab": args.user_vocab,
            "max_src_len": args.max_src_len,
            "max_tgt_len": args.max_tgt_len,
        },
    )

    train_loop(model, loader, args, device, wandb)

    ckpt_path = output_dir / "seq2seq.pt"
    torch.save({"model": model.state_dict(), "vocab_size": vocab_size}, ckpt_path)
    print("Saved checkpoint:", ckpt_path)
    wandb.finish()


if __name__ == "__main__":
    main()
