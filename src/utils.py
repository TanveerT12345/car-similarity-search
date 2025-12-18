# src/utils.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _collate_keep_meta(batch):
    # DataLoader normally tries to "stack" dicts by matching keys.
    # Our meta dicts can differ (some samples may not have class_name), so we keep metas as a list.
    imgs, labels, metas = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels, list(metas)


@torch.no_grad()
def embed_dataset(model, dataset, batch_size=128, device="cuda"):
    # Turns a dataset of images into:
    # - an embedding matrix [N, D] for FAISS indexing/search
    # - a parallel list of metadata (paths, labels, class names when available)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,                 # stable on Windows; avoids multiprocessing + collate headaches
        pin_memory=(device == "cuda"),
        collate_fn=_collate_keep_meta, # crucial: prevents KeyError from mismatched meta keys
    )

    all_embs = []
    all_metas = []

    for imgs, labels, metas in tqdm(loader, desc="Embedding"):
        imgs = imgs.to(device, non_blocking=True)

        # Forward pass: convert images -> embedding vectors
        emb = model(imgs)

        # L2-normalize so cosine similarity works consistently across indexing and querying
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        all_embs.append(emb.detach().cpu().numpy())
        all_metas.extend(metas)

    embs = np.concatenate(all_embs, axis=0).astype("float32")
    return embs, all_metas
