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

    ckpt = torch.load("checkpoints/best.pt", map_location=device, weights_only=True)
    model = EmbeddingNet(embed_dim=512).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Gallery: val + (optional) unlabeled test for a larger demo gallery
    val_ds = get_val_dataset()
    if use_unlabeled_test_for_gallery:
        test_ds = get_unlabeled_test_dataset()
        # concatenate by simple list wrapper
        from torch.utils.data import ConcatDataset
        gallery_ds = ConcatDataset([val_ds, test_ds])
    else:
        gallery_ds = val_ds

    embs, metas = embed_dataset(model, gallery_ds, batch_size=128, device=device)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    embs = embs.astype("float32")

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)

    faiss.write_index(index, "indexes/faiss.index")
    with open("indexes/meta.json", "w") as f:
        json.dump(metas, f)

    print("Saved indexes/faiss.index and indexes/meta.json")
    print("Gallery size:", len(metas), "dim:", d)

if __name__ == "__main__":
    main(use_unlabeled_test_for_gallery=False)
