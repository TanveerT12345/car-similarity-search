import os, json
import numpy as np
import torch
import faiss

from src.model import EmbeddingNet
from src.datasets import get_val_dataset, get_unlabeled_test_dataset
from src.utils import embed_dataset


def main(use_unlabeled_test_for_gallery: bool = True):
    os.makedirs("indexes", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the trained embedding model (must match the same embed_dim used in training/search)
    ckpt = torch.load("checkpoints/best.pt", map_location=device, weights_only=True)
    model = EmbeddingNet(embed_dim=512).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Choose what images go into the searchable gallery.
    # - val_ds is labeled (cleaner results: has class_name)
    # - unlabeled test makes the gallery bigger, but many items will show class_name="unknown"
    val_ds = get_val_dataset()
    if use_unlabeled_test_for_gallery:
        test_ds = get_unlabeled_test_dataset()
        from torch.utils.data import ConcatDataset
        gallery_ds = ConcatDataset([val_ds, test_ds])
    else:
        gallery_ds = val_ds

    # Embed every gallery image -> matrix [N, D] used to build the FAISS index
    embs, metas = embed_dataset(model, gallery_ds, batch_size=128, device=device)

    # Normalize so inner product behaves like cosine similarity (matches query normalization in search.py)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    embs = embs.astype("float32")

    # Build FAISS index:
    # IndexFlatIP = exact nearest neighbor search using inner product (simple + solid baseline)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)

    # Save index + metadata for search.py / app.py
    faiss.write_index(index, "indexes/faiss.index")
    with open("indexes/meta.json", "w") as f:
        json.dump(metas, f)

    print("Saved indexes/faiss.index and indexes/meta.json")
    print("Gallery size:", len(metas), "dim:", d)


if __name__ == "__main__":
    main(use_unlabeled_test_for_gallery=False)
