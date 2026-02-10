import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from init.codebook_init import residual_kmeans_init
from models.rqvae import RQVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Train RQ-VAE on Sentence-T5 embeddings.")
    parser.add_argument(
        "--embeddings",
        default="~/datasets/processed/beauty/sentence_t5_embeddings.npy",
        help="Path to sentence_t5_embeddings.npy",
    )
    parser.add_argument(
        "--item_ids",
        default="~/datasets/processed/beauty/sentence_t5_item_ids.json",
        help="Path to sentence_t5_item_ids.json",
    )
    parser.add_argument(
        "--output_dir",
        default="~/datasets/processed/beauty",
        help="Directory to save checkpoints and semantic ids",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--commit_cost", type=float, default=0.25)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--num_codebooks", type=int, default=3)
    parser.add_argument("--init_kmeans", action="store_true")
    parser.add_argument("--codebook_init_path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb_project", default="retiger")
    parser.add_argument("--wandb_run_name", default="")
    return parser.parse_args()


def resolve_collisions(
    item_ids: List[str],
    codes: np.ndarray,
) -> Dict[str, List[int]]:
    groups: Dict[Tuple[int, ...], List[str]] = {}
    for item_id, code in zip(item_ids, codes):
        key = tuple(int(x) for x in code.tolist())
        groups.setdefault(key, []).append(item_id)

    resolved: Dict[str, List[int]] = {}
    for key, ids in groups.items():
        if len(ids) == 1:
            resolved[ids[0]] = list(key)
            continue
        if len(ids) == 2:
            resolved[ids[0]] = list(key) + [0]
            resolved[ids[1]] = list(key) + [1]
            continue
        for idx, item_id in enumerate(ids):
            resolved[item_id] = list(key) + [idx]
        print(f"Warning: collision group size {len(ids)} for key {key}")

    return resolved


def save_mappings(
    output_dir: Path,
    item_ids: List[str],
    codes: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_collisions(item_ids, codes)

    semantic_id_to_item: Dict[str, List[str]] = {}
    for item_id, code in resolved.items():
        key = ",".join(str(x) for x in code)
        semantic_id_to_item.setdefault(key, []).append(item_id)

    (output_dir / "semantic_ids.json").write_text(
        json.dumps(resolved, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "semantic_id_to_item.json").write_text(
        json.dumps(semantic_id_to_item, ensure_ascii=False), encoding="utf-8"
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    embeddings_path = Path(args.embeddings).expanduser()
    item_ids_path = Path(args.item_ids).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not embeddings_path.exists():
        raise SystemExit(f"Embeddings file not found: {embeddings_path}")
    if not item_ids_path.exists():
        raise SystemExit(f"Item ids file not found: {item_ids_path}")

    embeddings = np.load(embeddings_path)
    item_ids = json.loads(item_ids_path.read_text(encoding="utf-8"))
    if embeddings.shape[0] != len(item_ids):
        raise ValueError("Embeddings and item ids size mismatch")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(embeddings).float().to(device)

    codebook_payload = None
    if args.codebook_init_path:
        payload = torch.load(Path(args.codebook_init_path).expanduser(), map_location=device)
        codebook_payload = payload
        if "codebooks" in payload:
            codebooks = payload["codebooks"]
            args.num_codebooks = int(payload.get("num_codebooks", codebooks.shape[0]))
            args.codebook_size = int(payload.get("codebook_size", codebooks.shape[1]))
            args.latent_dim = int(payload.get("latent_dim", codebooks.shape[2]))

    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = RQVAE(
        input_dim=x.shape[1],
        hidden_dims=[512, 256, 128],
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        num_codebooks=args.num_codebooks,
        commit_cost=args.commit_cost,
    ).to(device)

    try:
        import wandb
    except Exception as exc:
        raise SystemExit(
            "Missing dependency for logging. Install: wandb. "
            f"Original error: {exc}"
        )

    wandb_config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "commit_cost": args.commit_cost,
        "latent_dim": args.latent_dim,
        "codebook_size": args.codebook_size,
        "num_codebooks": args.num_codebooks,
        "init_kmeans": args.init_kmeans,
        "codebook_init_path": args.codebook_init_path,
    }
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or None,
        config=wandb_config,
    )

    if codebook_payload and "codebooks" in codebook_payload:
        model.quantizer.set_codebooks(codebook_payload["codebooks"].to(device))
    elif args.init_kmeans:
        with torch.no_grad():
            z_e = model.encoder(x)
        codebooks = residual_kmeans_init(
            z_e,
            codebook_size=args.codebook_size,
            num_codebooks=args.num_codebooks,
            iters=20,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        model.quantizer.set_codebooks(codebooks)

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    recon_loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_q = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, _, q_loss = model(batch)
            recon_loss = recon_loss_fn(recon, batch)
            loss = recon_loss + q_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
            total_recon += recon_loss.item() * batch.size(0)
            total_q += q_loss.item() * batch.size(0)
        avg_loss = total_loss / len(dataset)
        avg_recon = total_recon / len(dataset)
        avg_q = total_q / len(dataset)
        wandb.log(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "recon_loss": avg_recon,
                "quant_loss": avg_q,
            }
        )
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}: loss={avg_loss:.6f}, recon={avg_recon:.6f}, quant={avg_q:.6f}"
            )

    ckpt_path = output_dir / "rqvae.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    print("Saved checkpoint:", ckpt_path)

    model.eval()
    codes = model.encode_codes(x, batch_size=args.batch_size).numpy()
    save_mappings(output_dir, item_ids, codes)
    print("Saved semantic id mappings to:", output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
