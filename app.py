import json
import faiss
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.model import EmbeddingNet
from src.datasets import get_transform

@st.cache_resource
def load_system():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    index = faiss.read_index("indexes/faiss.index")
    meta = json.load(open("indexes/meta.json"))

    ckpt = torch.load("checkpoints/best.pt", map_location=device)
    model = EmbeddingNet(embed_dim=512).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tfm = get_transform(train=False)
    return device, model, tfm, index, meta

st.title("Car Similarity Search (Metric Learning + FAISS)")

uploaded = st.file_uploader("Upload a car image", type=["jpg","jpeg","png"])
k = st.slider("Top-K", min_value=1, max_value=20, value=10)

if uploaded:
    device, model, tfm, index, meta = load_system()

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Query", use_container_width=True)

    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        q = model(x).detach().cpu().numpy().astype("float32")
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

    scores, idxs = index.search(q, k)

    st.subheader("Results")
    cols = st.columns(5)
    for j, (i, s) in enumerate(zip(idxs[0], scores[0])):
        m = meta[int(i)]
        res_path = m["path"]
        cname = m.get("class_name", "unknown")
        try:
            res_img = Image.open(res_path).convert("RGB")
        except Exception:
            continue
        with cols[j % 5]:
            st.image(res_img, use_container_width=True)
            st.caption(f"{cname}\nscore={float(s):.3f}")
