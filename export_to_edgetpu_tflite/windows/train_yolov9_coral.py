"""
Entraînement YOLO pour déploiement Coral Edge TPU.

- Dataset YOLO exporté depuis CVAT.
- Séparation train/val étanche par session vidéo.
- Résolution fixe 416x416, identique à l'export Coral.
"""

from collections import defaultdict
from pathlib import Path
import random
import re
import shutil

from ultralytics import YOLO
import yaml


# =====================================================
# CONFIGURATION
# =====================================================

DATASET_DIR = Path(r"A:\Camion\data")
CLASSES = ["marche_avant", "creneau"]
YAML_PATH = DATASET_DIR / "data.yaml"

# 416 est le compromis choisi pour la caméra 640x480.
IMGSZ = 416

# Base plus compatible Coral que YOLOv8/YOLO11.
# Tester d'abord yolov9t, puis yolov9s si la précision est insuffisante.
MODEL_WEIGHTS = "yolov9t.pt"

VAL_RATIO = 0.2
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extraire_id_video(nom_fichier: str) -> str:
    """Extrait l'identifiant de session à partir du nom de fichier."""
    match = re.search(
        r"(\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{6})",
        nom_fichier,
    )
    if match:
        return match.group(1)

    match = re.search(
        r"(surveillance_\d{4}_\d{2}_\d{2}_\d{6})",
        nom_fichier,
    )
    if match:
        return match.group(1)

    stem = Path(nom_fichier).stem
    if "_frame_" in stem:
        return stem.split("_frame_")[0]

    return stem


def generer_data_yaml() -> None:
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASSES)},
    }

    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"[OK] data.yaml généré : {YAML_PATH}")


def organiser_dataset_par_video() -> bool:
    img_root = DATASET_DIR / "images"
    lbl_root = DATASET_DIR / "labels"

    train_img_dir = img_root / "train"
    val_img_dir = img_root / "val"
    train_lbl_dir = lbl_root / "train"
    val_lbl_dir = lbl_root / "val"

    for cache_file in lbl_root.rglob("*.cache"):
        try:
            cache_file.unlink()
            print(f"[OK] Cache supprimé : {cache_file.name}")
        except OSError:
            pass

    for split in ["train", "val"]:
        for folder, dest in [
            (img_root / split, img_root),
            (lbl_root / split, lbl_root),
        ]:
            if folder.exists():
                for f in folder.iterdir():
                    if f.is_file():
                        shutil.move(str(f), str(dest / f.name))

    images = [
        f
        for f in img_root.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
    ]

    if not images:
        print(f"[ERREUR] Aucune image trouvée dans {img_root}")
        return False

    groupes = defaultdict(list)
    for img in images:
        groupes[extraire_id_video(img.name)].append(img)

    sessions = sorted(groupes.keys())
    print(f"[OK] {len(sessions)} sessions détectées.")

    random.seed(42)
    random.shuffle(sessions)

    nb_val = max(1, int(len(sessions) * VAL_RATIO)) if len(sessions) > 1 else 0
    val_sessions = set(sessions[:nb_val])
    train_sessions = set(sessions[nb_val:])

    for d in [
        train_img_dir,
        val_img_dir,
        train_lbl_dir,
        val_lbl_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    nb_bg = 0
    nb_labels = 0

    for session, frames in groupes.items():
        is_val = session in val_sessions

        dest_img = val_img_dir if is_val else train_img_dir
        dest_lbl = val_lbl_dir if is_val else train_lbl_dir

        for img_path in frames:
            shutil.move(str(img_path), str(dest_img / img_path.name))

            src_label = lbl_root / f"{img_path.stem}.txt"
            dst_label = dest_lbl / f"{img_path.stem}.txt"

            if src_label.exists():
                shutil.move(str(src_label), str(dst_label))
                nb_labels += 1
            else:
                dst_label.touch()
                nb_bg += 1

    print(f"[OK] Images annotées : {nb_labels}")
    print(f"[OK] Images background : {nb_bg}")
    print(f"[OK] Sessions validation : {sorted(val_sessions)}")

    return True


def entrainer() -> None:
    generer_data_yaml()

    if not organiser_dataset_par_video():
        return

    model = YOLO(MODEL_WEIGHTS)

    results = model.train(
        data=str(YAML_PATH),
        epochs=120,
        imgsz=IMGSZ,
        batch=-1,
        patience=25,
        project="runs/detect",
        name="yolo_camions_edgetpu_416",
        device=0,
        workers=2,
        cache="disk",
        plots=True,
        # Régularisation utile si le dataset est petit.
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
    )

    print("\n[OK] Entraînement terminé.")
    print(f"Meilleurs poids : {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    entrainer()
