import json
import faiss
import numpy as np
import torch
from PIL import Image

from src.model import EmbeddingNet
from src.datasets import get_transform

def main(query_path: str, k: int = 10):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    index = faiss.read_index("indexes/faiss.index")
    meta = json.load(open("indexes/meta.json"))

    ckpt = torch.load("checkpoints/best.pt", map_location=device)
    model = EmbeddingNet(embed_dim=512).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tfm = get_transform(train=False)
    img = Image.open(query_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        q = model(x).detach().cpu().numpy().astype("float32")
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

    scores, idxs = index.search(q, k)
    for rank, (i, s) in enumerate(zip(idxs[0], scores[0]), 1):
        m = meta[int(i)]
        cname = m.get("class_name", "unknown")
        print(rank, f"score={float(s):.4f}", cname, m["path"])

if __name__ == "__main__":
    import sys
    main(sys.argv[1], k=10)
