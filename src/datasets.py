# src/datasets.py
import os
from typing import Optional, List, Any, Dict, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import loadmat


def get_transform(train: bool = True, img_size: int = 224):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class StanfordCarsDevkitDataset(Dataset):
    """
    Kaggle layout:
      root/
        cars_train/              (may contain nested cars_train/)
        cars_test/               (may contain nested cars_test/)
        car_devkit/devkit/       (cars_train_annos.mat, cars_test_annos.mat, cars_meta.mat)

    Important:
      - Some versions have UNLABELED test annotations (no class/class_ field).
      - This dataset supports:
          split="train" -> requires labels, will error if missing
          split="test"  -> allowed even if unlabeled; label will be -1 if missing
    """

    def __init__(self, root: str, split: str, transform=None, crop_bbox: bool = True):
        assert split in ("train", "test")
        self.root = root
        self.split = split
        self.transform = transform
        self.crop_bbox = crop_bbox

        # Image folder
        img_dir = os.path.join(root, f"cars_{split}")
        nested_img_dir = os.path.join(img_dir, f"cars_{split}")
        if os.path.isdir(nested_img_dir):
            img_dir = nested_img_dir
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Missing image folder: {img_dir}")

        # Devkit folder
        devkit_dir = os.path.join(root, "car_devkit", "devkit")
        if not os.path.isdir(devkit_dir):
            raise FileNotFoundError(f"Missing devkit folder: {devkit_dir}")

        annos_path = os.path.join(devkit_dir, f"cars_{split}_annos.mat")
        if not os.path.isfile(annos_path):
            raise FileNotFoundError(f"Missing annotation file: {annos_path}")

        meta_path = os.path.join(devkit_dir, "cars_meta.mat")
        if not os.path.isfile(meta_path):
            meta_path = None

        # Class names (optional)
        self.class_names = None
        if meta_path:
            meta = loadmat(meta_path, squeeze_me=True, struct_as_record=False)
            if "class_names" in meta:
                raw = meta["class_names"]
                self.class_names = [str(x).strip() for x in list(raw)]

        annos = loadmat(annos_path, squeeze_me=True, struct_as_record=False)
        if "annotations" not in annos:
            raise KeyError(f"'annotations' not found in {annos_path}. Keys={list(annos.keys())}")
        A = annos["annotations"]

        self.samples: List[Dict[str, Any]] = []
        for a in list(A):
            # filename
            if hasattr(a, "fname"):
                fname = str(getattr(a, "fname"))
            elif hasattr(a, "filename"):
                fname = str(getattr(a, "filename"))
            else:
                raise AttributeError(f"Annotation missing fname/filename. fields={getattr(a, '_fieldnames', None)}")

            # class label (may be missing in test)
            cls = -1
            if hasattr(a, "class"):
                cls = int(getattr(a, "class")) - 1
            elif hasattr(a, "class_"):
                cls = int(getattr(a, "class_")) - 1
            elif hasattr(a, "cls"):
                cls = int(getattr(a, "cls")) - 1
            else:
                # unlabeled test is OK; labeled train is NOT OK
                if split == "train":
                    raise AttributeError(
                        f"Train annotation missing class field. fields={getattr(a, '_fieldnames', None)}"
                    )

            # bbox
            x1 = int(getattr(a, "bbox_x1"))
            y1 = int(getattr(a, "bbox_y1"))
            x2 = int(getattr(a, "bbox_x2"))
            y2 = int(getattr(a, "bbox_y2"))

            path = os.path.join(img_dir, fname)
            self.samples.append({
                "path": path,
                "label": cls,
                "bbox": (x1, y1, x2, y2),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(s["path"]).convert("RGB")

        if self.crop_bbox:
            x1, y1, x2, y2 = s["bbox"]
            if x2 > x1 and y2 > y1:
                img = img.crop((x1, y1, x2, y2))

        if self.transform:
            img = self.transform(img)

        label = int(s["label"])
        meta = {"path": s["path"], "label": label}
        if self.class_names and label >= 0 and label < len(self.class_names):
            meta["class_name"] = self.class_names[label]
        return img, label, meta


def get_train_dataset(
    root: Optional[str] = None,
    img_size: int = 224,
    val_split: float = 0.1,
    seed: int = 42
):
    """
    Returns (train_ds, val_ds) split from LABELED TRAIN set.
    This is necessary because the Kaggle 'test' annotations may be unlabeled.
    """
    root = root or os.environ.get("CARS_DATA_ROOT")
    if not root:
        raise ValueError("Set CARS_DATA_ROOT env var to your dataset root path.")

    full_train_aug = StanfordCarsDevkitDataset(
        root=root,
        split="train",
        transform=get_transform(train=True, img_size=img_size),
        crop_bbox=True,
    )

    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    idx = np.arange(len(full_train_aug))
    rng.shuffle(idx)

    n_val = int(len(full_train_aug) * val_split)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = torch.utils.data.Subset(full_train_aug, train_idx.tolist())

    # Validation uses eval transform (no randomness)
    full_train_eval = StanfordCarsDevkitDataset(
        root=root,
        split="train",
        transform=get_transform(train=False, img_size=img_size),
        crop_bbox=True,
    )
    val_ds = torch.utils.data.Subset(full_train_eval, val_idx.tolist())

    return train_ds, val_ds


def get_val_dataset(root: Optional[str] = None, img_size: int = 224):
    _, val_ds = get_train_dataset(root=root, img_size=img_size)
    return val_ds


def get_unlabeled_test_dataset(root: Optional[str] = None, img_size: int = 224):
    """
    Optional: use for qualitative demo gallery (no metrics).
    """
    root = root or os.environ.get("CARS_DATA_ROOT")
    if not root:
        raise ValueError("Set CARS_DATA_ROOT env var to your dataset root path.")
    return StanfordCarsDevkitDataset(
        root=root,
        split="test",
        transform=get_transform(train=False, img_size=img_size),
        crop_bbox=True,
    )
