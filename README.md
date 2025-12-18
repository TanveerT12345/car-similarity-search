# Car Similarity Search (Metric Learning + FAISS)

Given a car photo, this project returns the **top-K most visually similar cars** using:
- a **metric learning embedding model** (ResNet-based)
- **FAISS** nearest-neighbor search over embedded images

**Scripts:**
- `train_embedding.py` – train + save `checkpoints/best.pt`
- `build_index.py` – build FAISS index + save `indexes/faiss.index` and `indexes/meta.json`
- `search.py` – CLI retrieval
- `app.py` – Streamlit demo

---

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