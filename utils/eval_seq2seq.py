import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from models.seq_to_seq import Seq2SeqTransformer
from train.train_seq2seq import (
    BOS_ID,
    CODE_OFFSET,
    EOS_ID,
    PAD_ID,
    build_semantic_map,
    build_user_sequences,
    hash_user,
    tokens_from_item,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate seq2seq model with Recall@5 and NDCG@5.")
    parser.add_argument("--train", default="~/datasets/processed/beauty/train.jsonl")
    parser.add_argument("--valid", default="~/datasets/processed/beauty/valid.jsonl")
    parser.add_argument("--test", default="~/datasets/processed/beauty/test.jsonl")
    parser.add_argument("--semantic_ids", default="~/datasets/processed/beauty/semantic_ids.json")
    parser.add_argument("--semantic_id_to_item", default="~/datasets/processed/beauty/semantic_id_to_item.json")
    parser.add_argument("--checkpoint", default="~/datasets/processed/beauty/seq2seq.pt")
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--max_src_len", type=int, default=128)
    parser.add_argument("--max_tgt_len", type=int, default=8)
    parser.add_argument("--user_vocab", type=int, default=2000)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_eval_targets(path: Path) -> Dict[str, Tuple[int, str]]:
    targets: Dict[str, Tuple[int, str]] = {}
    for rec in read_jsonl(path):
        user_id = rec["user_id"]
        timestamp = rec["timestamp"]
        item_id = rec["item_id"]
        prev = targets.get(user_id)
        if prev is None or timestamp >= prev[0]:
            targets[user_id] = (timestamp, item_id)
    return targets


def build_code_vocab(semantic_map: Dict[str, List[int]]) -> int:
    max_token = 0
    for tokens in semantic_map.values():
        if tokens:
            max_token = max(max_token, max(tokens))
    return max_token + 1


def beam_search_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    src_pad: torch.Tensor,
    max_len: int,
    beam_size: int,
    device: torch.device,
) -> List[List[int]]:
    beams: List[Tuple[List[int], float, bool]] = [([BOS_ID], 0.0, False)]

    for _ in range(max_len):
        new_beams: List[Tuple[List[int], float, bool]] = []
        for tokens, score, ended in beams:
            if ended:
                new_beams.append((tokens, score, ended))
                continue
            tgt = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_pad = tgt.eq(PAD_ID)
            logits = model(src, tgt, src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            topk = torch.topk(log_probs, beam_size, dim=-1)
            for k in range(topk.indices.size(1)):
                token = topk.indices[0, k].item()
                new_score = score + topk.values[0, k].item()
                new_tokens = tokens + [token]
                new_beams.append((new_tokens, new_score, token == EOS_ID))
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
        if all(ended for _, _, ended in beams):
            break

    return [tokens for tokens, _, _ in beams]


def tokens_to_item_id(tokens: List[int], semantic_to_item: Dict[str, List[str]]) -> str | None:
    code_tokens = [t - CODE_OFFSET for t in tokens]
    key = ",".join(str(t) for t in code_tokens)
    items = semantic_to_item.get(key)
    if not items:
        return None
    return items[0]


def predict_topk_items(
    model: Seq2SeqTransformer,
    semantic_map: Dict[str, List[int]],
    semantic_to_item: Dict[str, List[str]],
    user_id: str,
    history_items: List[str],
    user_vocab: int,
    max_src_len: int,
    max_tgt_len: int,
    beam_size: int,
    device: torch.device,
) -> List[str]:
    code_vocab = build_code_vocab(semantic_map)
    user_token = CODE_OFFSET + code_vocab + hash_user(user_id, user_vocab)
    history_tokens: List[int] = []
    for item_id in history_items:
        history_tokens.extend(tokens_from_item(item_id, semantic_map))
    src = [user_token] + history_tokens
    src = src[-max_src_len:]
    src_tensor = torch.tensor([src], dtype=torch.long, device=device)
    src_pad = src_tensor.eq(PAD_ID)
    beams = beam_search_decode(model, src_tensor, src_pad, max_tgt_len, beam_size, device)

    items: List[str] = []
    for seq in beams:
        decoded = []
        for token in seq[1:]:
            if token == EOS_ID:
                break
            decoded.append(token)
        if not decoded:
            continue
        item_id = tokens_to_item_id(decoded, semantic_to_item)
        if item_id and item_id not in items:
            items.append(item_id)
        if len(items) >= beam_size:
            break
    return items


def evaluate_split(
    split_targets: Dict[str, Tuple[int, str]],
    train_sequences: Dict[str, List[str]],
    semantic_map: Dict[str, List[int]],
    semantic_to_item: Dict[str, List[str]],
    model: Seq2SeqTransformer,
    args,
    device: torch.device,
) -> Tuple[float, float, int]:
    recall_sum = 0.0
    ndcg_sum = 0.0
    count = 0
    total = len(split_targets)
    for idx, (user_id, (_, target_item)) in enumerate(split_targets.items(), start=1):
        history_items = train_sequences.get(user_id)
        if not history_items:
            continue
        if target_item not in semantic_map:
            continue
        missing = False
        for item_id in history_items:
            if item_id not in semantic_map:
                missing = True
                break
        if missing:
            continue

        preds = predict_topk_items(
            model,
            semantic_map,
            semantic_to_item,
            user_id,
            history_items,
            args.user_vocab,
            args.max_src_len,
            args.max_tgt_len,
            args.beam_size,
            device,
        )
        count += 1
        if target_item in preds:
            recall_sum += 1.0
            rank = preds.index(target_item) + 1
            ndcg_sum += 1.0 / math.log2(rank + 1)
        if idx % 100 == 0:
            print(
                f"Eval progress: {idx}/{total} users, recall@5={recall_sum / max(count, 1):.6f}, "
                f"ndcg@5={ndcg_sum / max(count, 1):.6f}"
            )
    if count == 0:
        return 0.0, 0.0, 0
    return recall_sum / count, ndcg_sum / count, count


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    semantic_path = Path(args.semantic_ids).expanduser()
    semantic_to_item_path = Path(args.semantic_id_to_item).expanduser()
    train_path = Path(args.train).expanduser()
    valid_path = Path(args.valid).expanduser()
    test_path = Path(args.test).expanduser()
    ckpt_path = Path(args.checkpoint).expanduser()

    semantic_map = build_semantic_map(semantic_path)
    semantic_to_item = json.loads(semantic_to_item_path.read_text(encoding="utf-8"))
    train_sequences = build_user_sequences(train_path)

    vocab_size = CODE_OFFSET + build_code_vocab(semantic_map) + args.user_vocab
    ckpt = None
    if not args.random_init:
        ckpt = torch.load(ckpt_path, map_location=device)
        vocab_size = int(ckpt.get("vocab_size", 0)) or vocab_size

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
    if ckpt is not None:
        model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    valid_targets = load_eval_targets(valid_path)
    test_targets = load_eval_targets(test_path)

    valid_recall, valid_ndcg, valid_count = evaluate_split(
        valid_targets,
        train_sequences,
        semantic_map,
        semantic_to_item,
        model,
        args,
        device,
    )
    test_recall, test_ndcg, test_count = evaluate_split(
        test_targets,
        train_sequences,
        semantic_map,
        semantic_to_item,
        model,
        args,
        device,
    )

    print(f"Valid users: {valid_count}")
    print(f"Valid Recall@5: {valid_recall:.6f}")
    print(f"Valid NDCG@5: {valid_ndcg:.6f}")
    print(f"Test users: {test_count}")
    print(f"Test Recall@5: {test_recall:.6f}")
    print(f"Test NDCG@5: {test_ndcg:.6f}")


if __name__ == "__main__":
    main()
