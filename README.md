# Car Similarity Search (Metric Learning + FAISS)

Given a car photo, this project returns the **top-K most visually similar cars** using:
- a **metric learning embedding model** (ResNet-based)
- **FAISS** nearest-neighbor search over embedded images

## File Guide (What each file does)

**Top-level scripts**
- `train_embedding.py` — trains the metric learning model (Triplet Loss) on a train/val split, prints Recall@K + mAP@K each epoch, and saves the best checkpoint to `checkpoints/best.pt`.
- `build_index.py` — embeds a “gallery” dataset (usually the labeled validation set), builds a FAISS index for nearest-neighbor search, and saves:
  - `indexes/faiss.index` (vectors)
  - `indexes/meta.json` (paths + labels + optional class names)
- `search.py` — command-line query tool: loads `best.pt` + FAISS index, embeds a query image, retrieves top-K matches, and prints results with similarity scores.
- `app.py` — Streamlit UI: upload an image → embeds it → FAISS search → shows results in a grid; caches model/index for fast interaction.

**Core library code (`src/`)**
- `src/model.py` — defines `EmbeddingNet` (pretrained ResNet50 backbone + linear projection head) and outputs L2-normalized embeddings for cosine-style similarity.
- `src/datasets.py` — Stanford Cars loader (Kaggle + devkit `.mat`): handles nested folders, reads bounding boxes + labels when available, and creates a labeled train/val split (since some test annotations are unlabeled).
- `src/utils.py` — embedding pipeline utilities: batches inference over a dataset and returns `(embeddings, metadata)`; includes a custom `collate_fn` to avoid DataLoader errors when metadata keys differ.
- `src/metrics.py` — retrieval evaluation: computes similarity matrix from embeddings and reports Recall@K and mAP@K.


## Setup (Windows PowerShell)

```powershell
cd C:\Users\YOURNAME\car-similarity-search
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**(Optional) Confirm GPU is detected:**

```powershell
python -c "import torch; print('cuda?', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

---

## Dataset (Stanford Cars)

Set `CARS_DATA_ROOT` to the folder that contains:
- `cars_train/`
- `cars_test/`
- `car_devkit/`

**Example:**

```powershell
$env:CARS_DATA_ROOT="C:\datasets\stanford_cars"
```

**Note:** Some Kaggle layouts nest images like:
- `cars_train/cars_train/*.jpg`
- `cars_test/cars_test/*.jpg`

This project's loader handles the nested folder layout.

**(Optional) Quick dataset check:**

```powershell
python -c "from src.datasets import get_train_dataset; tr,va=get_train_dataset(); print('train',len(tr),'val',len(va)); x,y,m=tr[0]; print(x.shape,y,m['path'])"
```

---

## Train the Embedding Model

```powershell
python -u train_embedding.py
```

**Output:**
- `checkpoints/best.pt`

During training it prints retrieval metrics like Recall@K and mAP@K on a validation split.

---

## Build the FAISS Index

```powershell
python build_index.py
```

**Expected outputs:**
- `indexes/faiss.index`
- `indexes/meta.json`

---

## Search (CLI)

Replace the image path with any car image you want to query:

```powershell
python search.py "C:\datasets\stanford_cars\cars_train\cars_train\00001.jpg"
```

---

## Run the Demo App

```powershell
streamlit run app.py
```

---

## Notes

- **Do not commit large artifacts:** `checkpoints/`, `indexes/`, dataset folders
- If you see `ValueError: Set CARS_DATA_ROOT...`, re-run the `$env:CARS_DATA_ROOT=...` line in your current terminal
- Make sure your `.gitignore` excludes `checkpoints/`, `indexes/`, and `.venv/`