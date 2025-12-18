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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    # Load train/val split
    t0 = time.time()
    train_ds, val_ds = get_train_dataset(val_split=0.1, seed=42)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} loaded in {time.time()-t0:.1f}s", flush=True)

    # More stable settings for Windows/CUDA
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    print(f"[data] batches/epoch={len(train_loader)}", flush=True)

    # Smaller embedding helps stability + speed
    model = EmbeddingNet(embed_dim=256).to(device)

    # No miner (simpler and stable)
    loss_fn = TripletMarginLoss(margin=0.2, distance=distances.CosineSimilarity())

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_r10 = -1.0

    for epoch in range(1, 6):  # start with 5 epochs; increase later
        model.train()
        total = 0.0
        epoch_t0 = time.time()
        print(f"\n[train] epoch {epoch}/5", flush=True)

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

        # Faster eval subset (so you don’t wait forever)
        val_n = min(300, len(val_ds))
        val_subset = torch.utils.data.Subset(val_ds, range(val_n))
        print(f"[eval] on {val_n} images...", flush=True)
        metrics = evaluate_retrieval(model, val_subset, device=device, k_list=(1, 5, 10))
        print(f"[eval] {metrics}", flush=True)

        if metrics["R@10"] > best_r10:
            best_r10 = metrics["R@10"]
            torch.save({"model": model.state_dict(), "embed_dim": 256}, "checkpoints/best.pt")
            print("[ckpt] saved checkpoints/best.pt", flush=True)

    print("\n[done] best R@10 =", best_r10, flush=True)


if __name__ == "__main__":
    main()
