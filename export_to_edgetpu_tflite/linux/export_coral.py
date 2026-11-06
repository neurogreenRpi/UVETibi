"""
Export d'un modèle Ultralytics YOLO vers Google Coral Edge TPU.

À exécuter sous Linux / WSL2 x86_64, dans l'environnement coral_export.

Étapes :
1. Vérifications (poids, dataset, compilateur, mémoire).
2. Export Ultralytics : PyTorch -> ONNX -> TensorFlow -> TFLite INT8
   -> compilation Edge TPU.
3. Recherche du fichier *_edgetpu.tflite et copie dans exports_coral/.

Usage :
    python export_coral.py
    python export_coral.py --fraction 0.05
    python export_coral.py --imgsz 416 --fraction 0.1
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO


# =====================================================
# CONFIGURATION
# =====================================================

BEST_PT = Path(
    "/mnt/a/Camion/runs/detect/runs/detect/"
    "yolo_camions_edgetpu_416/weights/best.pt"
)

# YAML adapté à Linux (champ "path" en /mnt/a/...).
DATASET_YAML = Path("/mnt/a/Camion/data/data_linux.yaml")

# Doit être identique à l'entraînement.
DEFAULT_IMGSZ = 416

# Fraction du dataset utilisée pour la calibration INT8.
# 0.1 correspond à environ 300 images pour 3095 images de validation.
DEFAULT_FRACTION = 0.1

OUTPUT_DIR = Path("exports_coral")

# Mémoire disponible minimale conseillée avant de lancer l'export (en Go).
MIN_AVAILABLE_GB = 6.0


# =====================================================
# OUTILS
# =====================================================

def memoire_disponible_go() -> float | None:
    """Retourne la mémoire disponible en Go (Linux), ou None si inconnue."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for ligne in f:
                if ligne.startswith("MemAvailable:"):
                    return int(ligne.split()[1]) / 1024 / 1024
    except OSError:
        return None
    return None


def verifier_compilateur() -> bool:
    """Vérifie que edgetpu_compiler est installé et exécutable."""
    chemin = shutil.which("edgetpu_compiler")
    if chemin is None:
        print("[ERREUR] edgetpu_compiler est introuvable dans le PATH.")
        return False

    print(f"[OK] Compilateur : {chemin}")

    try:
        resultat = subprocess.run(
            ["edgetpu_compiler", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[ERREUR] Impossible d'exécuter le compilateur : {exc}")
        return False

    sortie = (resultat.stdout + resultat.stderr).strip()
    if resultat.returncode != 0:
        print(f"[ERREUR] Le compilateur renvoie une erreur :\n{sortie}")
        return False

    print(f"[OK] {sortie}")
    return True


def verifier_prerequis(imgsz: int, fraction: float) -> bool:
    ok = True

    if not BEST_PT.exists():
        print(f"[ERREUR] Poids introuvables : {BEST_PT}")
        ok = False

    if not DATASET_YAML.exists():
        print(f"[ERREUR] Dataset YAML introuvable : {DATASET_YAML}")
        ok = False

    if not 0.0 < fraction <= 1.0:
        print(f"[ERREUR] fraction doit être comprise entre 0 et 1 : {fraction}")
        ok = False

    if imgsz <= 0 or imgsz % 32 != 0:
        print(f"[ERREUR] imgsz doit être un multiple de 32 : {imgsz}")
        ok = False

    if not verifier_compilateur():
        ok = False

    memoire = memoire_disponible_go()
    if memoire is None:
        print("[INFO] Mémoire disponible non déterminée.")
    else:
        print(f"[INFO] Mémoire disponible : {memoire:.1f} Go")
        if memoire < MIN_AVAILABLE_GB:
            print(
                f"[ATTENTION] Moins de {MIN_AVAILABLE_GB:.0f} Go disponibles. "
                "Le processus risque d'être tué (Killed). "
                "Augmente la mémoire dans .wslconfig ou réduis --fraction."
            )

    return ok


def trouver_modele_edgetpu(dossier_poids: Path) -> Path | None:
    """Cherche le fichier *_edgetpu.tflite le plus récent près des poids."""
    candidats = sorted(
        dossier_poids.rglob("*_edgetpu.tflite"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidats[0] if candidats else None


# =====================================================
# EXPORT
# =====================================================

def exporter(imgsz: int, fraction: float) -> int:
    if not verifier_prerequis(imgsz, fraction):
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Configuration de l'export")
    print(f" - Poids    : {BEST_PT}")
    print(f" - Dataset  : {DATASET_YAML}")
    print(f" - imgsz    : {imgsz}")
    print(f" - fraction : {fraction}")
    print(f" - Sortie   : {OUTPUT_DIR.resolve()}\n")

    model = YOLO(str(BEST_PT))

    # Les fichiers d'export sont écrits près des poids d'origine.
    # La quantification INT8 est activée automatiquement pour edgetpu.
    try:
        exporte = model.export(
            format="edgetpu",
            imgsz=imgsz,
            data=str(DATASET_YAML),
            fraction=fraction,
            device="cpu",
        )
    except Exception as exc:
        print(f"\n[ERREUR] Échec de l'export : {exc}")
        print("Consulte le journal complet ci-dessus.")
        return 2

    print(f"\n[INFO] Chemin retourné par Ultralytics : {exporte}")

    modele = None
    if exporte:
        chemin = Path(str(exporte))
        if chemin.is_file() and chemin.name.endswith("_edgetpu.tflite"):
            modele = chemin

    if modele is None:
        modele = trouver_modele_edgetpu(BEST_PT.parent)

    if modele is None:
        print("[ERREUR] Aucun fichier *_edgetpu.tflite trouvé.")
        print(
            "Un fichier INT8 non compilé peut exister dans "
            f"{BEST_PT.parent}. Compile-le manuellement avec "
            "edgetpu_compiler."
        )
        return 3

    destination = OUTPUT_DIR / modele.name
    shutil.copy2(modele, destination)

    print("\n[OK] Export terminé.")
    print(f" - Modèle compilé : {modele}")
    print(f" - Copie          : {destination.resolve()}")

    journal = modele.with_suffix(".log")
    if journal.exists():
        print(f" - Rapport compilateur : {journal}")
        shutil.copy2(journal, OUTPUT_DIR / journal.name)

    print("\nÀ vérifier dans le rapport du compilateur :")
    print(" - nombre d'opérations exécutées sur Edge TPU ;")
    print(" - nombre d'opérations laissées sur CPU.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export YOLO vers Coral Edge TPU."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Résolution d'export (identique à l'entraînement).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=DEFAULT_FRACTION,
        help="Fraction du dataset utilisée pour la calibration INT8.",
    )
    args = parser.parse_args()

    return exporter(args.imgsz, args.fraction)


if __name__ == "__main__":
    sys.exit(main())
