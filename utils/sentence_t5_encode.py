import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text as text  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "Missing dependencies. Install: numpy, tensorflow, tensorflow_hub, tensorflow_text. "
        f"Original error: {exc}"
    )


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _extract_tensor(output):
    if isinstance(output, dict):
        if "outputs" in output:
            return output["outputs"]
        if "pooled_output" in output:
            return output["pooled_output"]
        if "embedding" in output:
            return output["embedding"]
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("Model output list is empty")
        return output[0]
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Encode item texts with Sentence-T5.")
    parser.add_argument(
        "--input",
        default="~/datasets/processed/beauty/item_text.jsonl",
        help="Path to item_text.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="~/datasets/processed/beauty",
        help="Directory to save embeddings and item ids",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--output_format",
        choices=["npy", "pt"],
        default="npy",
        help="Output format: npy or pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    hub_url = "https://www.kaggle.com/models/google/sentence-t5/TensorFlow2/st5-base/1"
    encoder = hub.KerasLayer(hub_url)

    item_ids = []
    texts = []
    for rec in read_jsonl(input_path):
        item_id = rec.get("item_id")
        text_val = rec.get("text", "")
        if not item_id:
            continue
        text_val = text_val.strip()
        if not text_val:
            continue
        item_ids.append(item_id)
        texts.append(text_val)

    if not texts:
        raise ValueError("No input texts found in item_text.jsonl")

    batch_size = args.batch_size
    embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_tensor = tf.constant(batch)
        batch_emb = encoder(batch_tensor)
        batch_tensor_out = _extract_tensor(batch_emb)
        embeds.append(batch_tensor_out.numpy())

    embeddings = np.vstack(embeds)

    embeddings = embeddings.astype("float32")
    if args.output_format == "npy":
        np.save(output_dir / "sentence_t5_embeddings.npy", embeddings)
        (output_dir / "sentence_t5_item_ids.json").write_text(
            json.dumps(item_ids, ensure_ascii=False),
            encoding="utf-8",
        )
        print("Saved embeddings:", output_dir / "sentence_t5_embeddings.npy")
        print("Saved item ids:", output_dir / "sentence_t5_item_ids.json")
    else:
        try:
            import torch
        except Exception as exc:
            raise SystemExit(
                "Missing dependency for pt output. Install: torch. "
                f"Original error: {exc}"
            )
        payload = {
            "item_ids": item_ids,
            "embeddings": torch.from_numpy(embeddings),
        }
        torch.save(payload, output_dir / "sentence_t5_embeddings.pt")
        print("Saved embeddings:", output_dir / "sentence_t5_embeddings.pt")
    print("Shape:", embeddings.shape)


if __name__ == "__main__":
    main()
