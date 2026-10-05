"""
Prépare le dataset Kaggle "Vehicle Detection - 8 Classes" (format YOLO) :
- vérifie la correspondance image/label
- split train/val (85/15 par défaut)
- génère data.yaml prêt pour l'entraînement YOLO (local, Colab ou Kaggle)

Usage:
    python prepare_dataset.py /chemin/vers/dataset_extrait
    (le dossier doit contenir train/images, train/labels, train/labels/classes.txt)

Produit un dossier `dataset_yolo/` avec :
    dataset_yolo/
        images/train, images/val
        labels/train, labels/val
        data.yaml
"""

import random
import shutil
import sys
from pathlib import Path

VAL_RATIO = 0.15
SEED = 42


def load_classes(labels_dir: Path) -> list[str]:
    classes_file = labels_dir / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(
            f"classes.txt introuvable dans {labels_dir} — "
            "impossible de connaître le nom des classes."
        )
    text = classes_file.read_text(encoding="utf-8").replace("\r", "")
    return [c.strip() for c in text.splitlines() if c.strip()]


def find_pairs(images_dir: Path, labels_dir: Path):
    stems_img = {p.stem: p for p in images_dir.glob("*.jpg")}
    stems_img.update({p.stem: p for p in images_dir.glob("*.png")})
    pairs = []
    for stem, img_path in stems_img.items():
        label_path = labels_dir / f"{stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def main(src_root: str, out_root: str = "dataset_yolo"):
    src_root = Path(src_root)
    images_dir = src_root / "train" / "images"
    labels_dir = src_root / "train" / "labels"
    out_root = Path(out_root)

    class_names = load_classes(labels_dir)
    print(f"{len(class_names)} classes : {class_names}")

    pairs = find_pairs(images_dir, labels_dir)
    print(f"{len(pairs)} paires image/label valides trouvées")

    random.seed(SEED)
    random.shuffle(pairs)
    n_val = int(len(pairs) * VAL_RATIO)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"Split : {len(train_pairs)} train / {len(val_pairs)} val")

    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_path, label_path in split_pairs:
            shutil.copy2(img_path, out_root / "images" / split / img_path.name)
            shutil.copy2(label_path, out_root / "labels" / split / label_path.name)

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "path: " + str(out_root.resolve()) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n",
        encoding="utf-8",
    )
    print(f"data.yaml écrit dans {data_yaml}")
    print("Dataset prêt pour l'entraînement.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prepare_dataset.py /chemin/vers/dataset_extrait")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "dataset_yolo")
