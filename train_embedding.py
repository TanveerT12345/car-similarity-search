import os
import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner
from pytorch_metric_learning import distances

from src.datasets import get_train_dataset, get_val_dataset
from src.model import EmbeddingNet
from src.metrics import evaluate_retrieval

def main():
    os.makedirs("checkpoints", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds, val_ds = get_train_dataset(val_split=0.1, seed=42)

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,   # Mac stability
        drop_last=True
    )

    model = EmbeddingNet(embed_dim=512).to(device)

    miner = MultiSimilarityMiner(epsilon=0.1)
    loss_fn = TripletMarginLoss(margin=0.2, distance=distances.CosineSimilarity())

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_r10 = -1.0
    for epoch in range(1, 11):
        model.train()
        total = 0.0

        for imgs, labels, _meta in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            emb = model(imgs)
            hard = miner(emb, labels)
            loss = loss_fn(emb, labels, hard)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        val_subset = torch.utils.data.Subset(val_ds, range(min(1000, len(val_ds))))
        metrics = evaluate_retrieval(model, val_subset, device=device, k_list=(1,5,10))
        print(f"Epoch {epoch} | loss={total/len(train_loader):.4f} | {metrics}")

        if metrics["R@10"] > best_r10:
            best_r10 = metrics["R@10"]
            torch.save({"model": model.state_dict()}, "checkpoints/best.pt")
            print("Saved checkpoints/best.pt")

if __name__ == "__main__":
    main()
