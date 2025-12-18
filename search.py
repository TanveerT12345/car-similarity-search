import json
import faiss
import numpy as np
import torch
from PIL import Image

from src.model import EmbeddingNet
from src.datasets import get_transform


def main(query_path: str, k: int = 10):
    # 1) Load the retrieval artifacts created by build_index.py:
    #    - FAISS index = fast nearest-neighbor search over embedding vectors
    #    - meta.json   = maps FAISS row -> image path + optional class name/label
    index = faiss.read_index("indexes/faiss.index")
    meta = json.load(open("indexes/meta.json"))

    # 2) Load the trained embedding model (same architecture as training)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load("checkpoints/best.pt", map_location=device)
    model = EmbeddingNet(embed_dim=512).to(device)   # embed_dim must match training
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 3) Preprocess the query image exactly like evaluation/indexing
    tfm = get_transform(train=False)
    img = Image.open(query_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

    # 4) Embed the query image -> 1 vector, then L2-normalize for cosine-style similarity
    with torch.no_grad():
        q = model(x).detach().cpu().numpy().astype("float32")
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

    # 5) Retrieve top-k nearest neighbors from FAISS and print results
    scores, idxs = index.search(q, k)
    for rank, (i, s) in enumerate(zip(idxs[0], scores[0]), 1):
        m = meta[int(i)]
        cname = m.get("class_name", "unknown")  # unknown usually means unlabeled gallery item
        print(rank, f"score={float(s):.4f}", cname, m["path"])


if __name__ == "__main__":
    import sys
    main(sys.argv[1], k=10)
