import json
import sys
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from models.seq_to_seq import Seq2SeqTransformer
from train.train_seq2seq import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    build_examples,
    build_semantic_map,
    build_user_sequences,
    build_vocab_size,
    collate_fn,
    run_inference,
)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    tmp_dir = Path(__file__).resolve().parent / "_tmp_seq2seq"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    semantic_ids = {
        "i1": [0, 1, 2],
        "i2": [3, 4, 5],
        "i3": [6, 7, 8],
    }
    semantic_path = tmp_dir / "semantic_ids.json"
    semantic_path.write_text(json.dumps(semantic_ids), encoding="utf-8")

    train_rows = [
        {"user_id": "u1", "item_id": "i1", "timestamp": 1},
        {"user_id": "u1", "item_id": "i2", "timestamp": 2},
        {"user_id": "u1", "item_id": "i3", "timestamp": 3},
    ]
    train_path = tmp_dir / "train.jsonl"
    write_jsonl(train_path, train_rows)

    semantic_map = build_semantic_map(semantic_path)
    sequences = build_user_sequences(train_path)
    examples = build_examples(sequences, semantic_map)

    dataset = [
        {
            "src": torch.tensor([3 + t for t in semantic_ids["i1"]]),
            "tgt_in": torch.tensor([BOS_ID, 3 + semantic_ids["i2"][0]]),
            "tgt_out": torch.tensor([3 + semantic_ids["i2"][0], EOS_ID]),
        }
    ]
    batch = collate_fn(dataset)
    assert batch["src"].shape[0] == 1

    vocab_size = build_vocab_size(semantic_map, user_vocab=10)
    model = Seq2SeqTransformer(vocab_size=vocab_size, pad_id=PAD_ID, nhead=4)
    model.eval()

    decoded = run_inference(
        model,
        semantic_map,
        user_id="u1",
        history_items=["i1"],
        user_vocab=10,
        max_src_len=16,
        max_tgt_len=4,
        device=torch.device("cpu"),
    )
    assert decoded[0] == BOS_ID
    assert len(decoded) >= 2

    print("OK")


if __name__ == "__main__":
    main()
