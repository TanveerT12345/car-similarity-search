# train_embedding.py
import os
import time
import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning import distances

from src.datasets import get_train_dataset
from src.model import EmbeddingNet
from src.metrics import evaluate_retrieval


def main():
    os.makedirs("checkpoints", exist_ok=True)

    # Use GPU if available (major speedup); print immediately so you know what hardware is used
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    # Load a train/val split from the labeled training set so we can measure retrieval performance on held-out data
    t0 = time.time()
    train_ds, val_ds = get_train_dataset(val_split=0.1, seed=42)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} loaded in {time.time()-t0:.1f}s", flush=True)

    # DataLoader settings chosen for Windows/CUDA stability
    # - smaller batch_size reduces GPU memory pressure
    # - num_workers=0 avoids multiprocessing issues on Windows
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    print(f"[data] batches/epoch={len(train_loader)}", flush=True)

    # Model learns an embedding vector per image; smaller embed_dim is faster and more stable to train
    model = EmbeddingNet(embed_dim=256).to(device)

    # Metric learning objective: push same-class embeddings closer than different-class embeddings
    # We use cosine distance because later retrieval uses normalized vectors (cosine-style similarity)
    loss_fn = TripletMarginLoss(margin=0.2, distance=distances.CosineSimilarity())

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Mixed precision speeds up training and reduces memory usage on GPU (when CUDA is enabled)
    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Track best checkpoint by Recall@10 (retrieval-focused metric)
    best_r10 = -1.0

    # Start with a small number of epochs for iteration; increase once things look good
    for epoch in range(1, 6):
        model.train()
        total = 0.0
        epoch_t0 = time.time()
        print(f"\n[train] epoch {epoch}/5", flush=True)

        # Train loop with periodic logging so you know it’s progressing
        step_t0 = time.time()
        for step, (imgs, labels, _meta) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                emb = model(imgs)
                loss = loss_fn(emb, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += float(loss.item())

            if step == 1 or step % 50 == 0:
                avg_loss = total / step
                print(f"[train] step {step}/{len(train_loader)} avg_loss={avg_loss:.4f} (+{time.time()-step_t0:.1f}s)", flush=True)
                step_t0 = time.time()

        print(f"[train] epoch {epoch} done avg_loss={total/max(1,len(train_loader)):.4f} time={time.time()-epoch_t0:.1f}s", flush=True)

        # Evaluate retrieval on a small validation subset for speed (Recall@K + mAP@K)
        val_n = min(300, len(val_ds))
        val_subset = torch.utils.data.Subset(val_ds, range(val_n))
        print(f"[eval] on {val_n} images...", flush=True)
        metrics = evaluate_retrieval(model, val_subset, device=device, k_list=(1, 5, 10))
        print(f"[eval] {metrics}", flush=True)

        # Save best model checkpoint; store embed_dim so build_index.py/search.py can match it safely
        if metrics["R@10"] > best_r10:
            best_r10 = metrics["R@10"]
            torch.save({"model": model.state_dict(), "embed_dim": 256}, "checkpoints/best.pt")
            print("[ckpt] saved checkpoints/best.pt", flush=True)

    print("\n[done] best R@10 =", best_r10, flush=True)


if __name__ == "__main__":
    main()
