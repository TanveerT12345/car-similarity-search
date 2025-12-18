import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def embed_dataset(model, dataset, batch_size=128, num_workers=2, device="cpu"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embs = []
    metas = []

    model.eval()
    with torch.no_grad():
        for imgs, labels, meta in tqdm(loader, desc="Embedding"):
            imgs = imgs.to(device)
            z = model(imgs).detach().cpu().numpy().astype("float32")
            embs.append(z)

            if isinstance(meta, dict):
                bs = len(labels)
                for i in range(bs):
                    item = {k: meta[k][i] for k in meta}
                    item["label"] = int(labels[i])
                    metas.append(item)
            else:
                for i in range(len(labels)):
                    metas.append({"label": int(labels[i])})

    embs = np.concatenate(embs, axis=0)
    return embs, metas
