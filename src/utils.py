# src/utils.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _collate_keep_meta(batch):
    """
    batch is a list of (img_tensor, label_int, meta_dict).
    We stack images/labels, but keep meta as a Python list so dict keys don't have to match.
    """
    imgs, labels, metas = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels, list(metas)


@torch.no_grad()
def embed_dataset(model, dataset, batch_size=128, device="cuda"):
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,                 # Windows stability + avoids worker collate issues
        pin_memory=(device == "cuda"),
        collate_fn=_collate_keep_meta, # <-- key fix
    )

    all_embs = []
    all_metas = []

    for imgs, labels, metas in tqdm(loader, desc="Embedding"):
        imgs = imgs.to(device, non_blocking=True)
        emb = model(imgs)

        # Normalize for cosine similarity retrieval
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        all_embs.append(emb.detach().cpu().numpy())
        all_metas.extend(metas)

    embs = np.concatenate(all_embs, axis=0)
    return embs.astype("float32"), all_metas

