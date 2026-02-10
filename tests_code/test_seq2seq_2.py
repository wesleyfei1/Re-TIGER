import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from models.seq_to_seq import Seq2SeqTransformer
from train.train_seq2seq import (
    BOS_ID,
    CODE_OFFSET,
    EOS_ID,
    build_semantic_map,
    build_user_sequences,
    build_vocab_size,
    run_inference,
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_item_text_map(path: Path) -> Dict[str, str]:
    mapping = {}
    for rec in read_jsonl(path):
        item_id = rec.get("item_id")
        text = rec.get("text", "")
        if item_id:
            mapping[item_id] = text
    return mapping


def pick_sample(sequences: Dict[str, List[str]]) -> tuple[str, List[str], str]:
    for user_id, items in sequences.items():
        if len(items) >= 2:
            history = items[:-1]
            target = items[-1]
            return user_id, history, target
    raise ValueError("No user with at least 2 interactions")


def tokens_to_item_id(tokens: List[int], semantic_to_item: Dict[str, List[str]]) -> str | None:
    if not tokens:
        return None
    code_tokens = [t - CODE_OFFSET for t in tokens]
    key = ",".join(str(t) for t in code_tokens)
    items = semantic_to_item.get(key)
    if not items:
        return None
    return items[0]


def main():
    data_dir = Path("~/datasets/processed/beauty").expanduser()
    train_path = data_dir / "train.jsonl"
    semantic_path = data_dir / "semantic_ids.json"
    semantic_to_item_path = data_dir / "semantic_id_to_item.json"
    item_text_path = data_dir / "item_text.jsonl"

    semantic_map = build_semantic_map(semantic_path)
    semantic_to_item = json.loads(semantic_to_item_path.read_text(encoding="utf-8"))
    item_text = build_item_text_map(item_text_path)

    sequences = build_user_sequences(train_path)
    user_id, history_items, target_item = pick_sample(sequences)

    vocab_size = build_vocab_size(semantic_map, user_vocab=2000)
    model = Seq2SeqTransformer(vocab_size=vocab_size, pad_id=0, nhead=4)
    model.eval()

    decoded = run_inference(
        model,
        semantic_map,
        user_id=user_id,
        history_items=history_items,
        user_vocab=2000,
        max_src_len=128,
        max_tgt_len=8,
        device=torch.device("cpu"),
    )

    decoded_tokens = []
    for token in decoded[1:]:
        if token == EOS_ID:
            break
        decoded_tokens.append(token)

    pred_item = tokens_to_item_id(decoded_tokens, semantic_to_item)
    pred_text = item_text.get(pred_item, "") if pred_item else ""
    target_text = item_text.get(target_item, "")

    print("User:", user_id)
    print("History length:", len(history_items))
    print("Decoded tokens:", decoded)
    print("Pred item_id:", pred_item)
    print("Pred item text:", pred_text)
    print("Target item_id:", target_item)
    print("Target item text:", target_text)


if __name__ == "__main__":
    main()
