import numpy as np
from src.utils import embed_dataset


def recall_at_k(sim, labels, k):
    # Recall@K: for each query image, check whether at least one of the top-K retrieved
    # images has the same label/class as the query.
    #
    # Why it's useful: it answers “does the system return a correct match somewhere in the top-K?”
    np.fill_diagonal(sim, -1e9)  # ignore self-match (an image is always closest to itself)
    idx = np.argpartition(-sim, kth=k-1, axis=1)[:, :k]  # fast top-K without full sort
    hits = (labels[idx] == labels[:, None]).any(axis=1)
    return float(hits.mean())


def map_at_k(sim, labels, k):
    # mAP@K (mean Average Precision at K): rewards ranking quality.
    # If the correct-class images appear earlier in the top-K list, mAP@K is higher.
    #
    # Why it's useful: it answers “how well are correct matches ranked near the top?”
    np.fill_diagonal(sim, -1e9)  # ignore self-match
    order = np.argsort(-sim, axis=1)[:, :k]  # full sort to compute ranked precision
    aps = []

    for i in range(sim.shape[0]):
        y = labels[i]
        retrieved = labels[order[i]]
        rel = (retrieved == y).astype(np.int32)

        # If there are no relevant items in top-K, AP for this query is 0
        if rel.sum() == 0:
            aps.append(0.0)
            continue

        # Average Precision = average of precision@j at each relevant position j
        c = 0
        precs = []
        for j in range(k):
            if rel[j]:
                c += 1
                precs.append(c / (j + 1))
        aps.append(float(np.mean(precs)))

    return float(np.mean(aps))


def evaluate_retrieval(model, dataset, device="cpu", k_list=(1, 5, 10)):
    # End-to-end retrieval evaluation:
    # 1) embed all images in the dataset
    # 2) compute pairwise cosine similarities
    # 3) report Recall@K and mAP@K
    embs, metas = embed_dataset(model, dataset, batch_size=128, device=device)
    labels = np.array([m["label"] for m in metas], dtype=np.int32)

    # Normalize embeddings (safety) so cosine similarity is stable
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    # Cosine similarity matrix (N x N). Higher = more similar.
    sim = embs @ embs.T

    out = {}
    for k in k_list:
        out[f"R@{k}"] = recall_at_k(sim.copy(), labels, k)

    # Report a single mAP@K at the largest K requested (common convention)
    out[f"mAP@{k_list[-1]}"] = map_at_k(sim.copy(), labels, k_list[-1])
    return out
