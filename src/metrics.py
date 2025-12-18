import numpy as np
from src.utils import embed_dataset

def recall_at_k(sim, labels, k):
    np.fill_diagonal(sim, -1e9)
    idx = np.argpartition(-sim, kth=k-1, axis=1)[:, :k]
    hits = (labels[idx] == labels[:, None]).any(axis=1)
    return float(hits.mean())

def map_at_k(sim, labels, k):
    np.fill_diagonal(sim, -1e9)
    order = np.argsort(-sim, axis=1)[:, :k]
    aps = []
    for i in range(sim.shape[0]):
        y = labels[i]
        retrieved = labels[order[i]]
        rel = (retrieved == y).astype(np.int32)
        if rel.sum() == 0:
            aps.append(0.0)
            continue
        c = 0
        precs = []
        for j in range(k):
            if rel[j]:
                c += 1
                precs.append(c / (j + 1))
        aps.append(float(np.mean(precs)))
    return float(np.mean(aps))

def evaluate_retrieval(model, dataset, device="cpu", k_list=(1,5,10)):
    embs, metas = embed_dataset(model, dataset, batch_size=128, device=device)
    labels = np.array([m["label"] for m in metas], dtype=np.int32)

    # normalize (safety)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    sim = embs @ embs.T

    out = {}
    for k in k_list:
        out[f"R@{k}"] = recall_at_k(sim.copy(), labels, k)
    out[f"mAP@{k_list[-1]}"] = map_at_k(sim.copy(), labels, k_list[-1])
    return out
