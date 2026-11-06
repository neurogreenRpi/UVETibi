"""
Export d'un modele Ultralytics YOLO vers Google Coral Edge TPU.

A executer sous Linux / WSL2 x86_64, dans l'environnement coral_export.

Etapes :
1. Verifications (poids, dataset, compilateur, memoire).
2. Export Ultralytics : PyTorch -> ONNX -> TensorFlow -> TFLite INT8
   -> compilation Edge TPU (avec le compilateur candidat, via une enveloppe).
3. Recherche du fichier *_edgetpu.tflite et copie dans exports_coral/.

Usage :
    python export_coral.py
    python export_coral.py --fraction 0.05
    python export_coral.py --imgsz 416 --fraction 0.1
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

# =====================================================
# CONFIGURATION
# =====================================================

BEST_PT = Path(
    "/mnt/a/best.pt"
)

# YAML adapte a Linux (champ "path" en /mnt/a/...).
DATASET_YAML = Path("/mnt/a/Camion/data/data_linux.yaml")

# Doit etre identique a la resolution retenue pour le projet.
DEFAULT_IMGSZ = 640

# Fraction du dataset utilisee pour la calibration INT8.
DEFAULT_FRACTION = 0.1

OUTPUT_DIR = Path("exports_coral")

# Memoire disponible minimale conseillee avant de lancer l'export (en Go).
MIN_AVAILABLE_GB = 6.0

# Compilateur candidat (version 15) : dossier, chargeur et executable.
COMPILER_DIR = Path("/root/tools/coral-compiler-candidate")
LOADER = COMPILER_DIR / "ld-linux-x86-64.so.2"
COMPILER = COMPILER_DIR / "edgetpu_compiler"

# Dossier de l'enveloppe "edgetpu_compiler" placee en tete du PATH.
WRAPPER_DIR = Path("/root/tools/bin")

# =====================================================
# OUTILS
# =====================================================


def memoire_disponible_go() -> float | None:
    """Retourne la memoire disponible en Go (Linux), ou None si inconnue."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for ligne in f:
                if ligne.startswith("MemAvailable:"):
                    return int(ligne.split()[1]) / 1024 / 1024
    except OSError:
        return None
    return None


def installer_enveloppe() -> bool:
    """Cree /root/tools/bin/edgetpu_compiler et le place en tete du PATH.

    Ultralytics appelle simplement "edgetpu_compiler" : cette enveloppe
    redirige l'appel vers le compilateur candidat et ses bibliotheques.
    """
    enveloppe = WRAPPER_DIR / "edgetpu_compiler"
    contenu = (
        "#!/usr/bin/env bash\n"
        f'DIR="{COMPILER_DIR}"\n'
        'exec "$DIR/ld-linux-x86-64.so.2" --library-path "$DIR" '
        '"$DIR/edgetpu_compiler" "$@"\n'
    )
    try:
        WRAPPER_DIR.mkdir(parents=True, exist_ok=True)
        enveloppe.write_text(contenu, encoding="utf-8", newline="\n")
        enveloppe.chmod(0o755)
    except OSError as exc:
        print(f"[ERREUR] Impossible de creer l'enveloppe {enveloppe} : {exc}")
        return False

    os.environ["PATH"] = f"{WRAPPER_DIR}:{os.environ.get('PATH', '')}"
    print(f"[OK] Enveloppe installee : {enveloppe}")
    return True


def verifier_compilateur() -> bool:
    """Verifie le compilateur candidat et l'utilise via l'enveloppe."""
    for fichier in (LOADER, COMPILER):
        if not fichier.is_file():
            print(f"[ERREUR] Fichier introuvable : {fichier}")
            return False

    print(f"[OK] Compilateur : {COMPILER}")

    # Test direct (chargeur + bibliotheques du dossier candidat).
    try:
        resultat = subprocess.run(
            [str(LOADER), "--library-path", str(COMPILER_DIR),
             str(COMPILER), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[ERREUR] Impossible d'executer le compilateur : {exc}")
        return False

    sortie = (resultat.stdout + resultat.stderr).strip()
    if resultat.returncode != 0:
        print(f"[ERREUR] Le compilateur renvoie une erreur :\n{sortie}")
        return False
    print(f"[OK] {sortie}")

    # Enveloppe + verification que "edgetpu_compiler" du PATH est le bon.
    if not installer_enveloppe():
        return False

    trouve = shutil.which("edgetpu_compiler")
    if trouve != str(WRAPPER_DIR / "edgetpu_compiler"):
        print(f"[ERREUR] Le PATH pointe vers un autre compilateur : {trouve}")
        return False

    try:
        via_path = subprocess.run(
            ["edgetpu_compiler", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[ERREUR] Enveloppe inutilisable : {exc}")
        return False

    sortie_path = (via_path.stdout + via_path.stderr).strip()
    if via_path.returncode != 0 or sortie_path != sortie:
        print(f"[ERREUR] Version differente via le PATH :\n{sortie_path}")
        return False

    print("[OK] Ultralytics utilisera ce compilateur (PATH verifie).")
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
        print(f"[ERREUR] fraction doit etre comprise entre 0 et 1 : {fraction}")
        ok = False

    if imgsz <= 0 or imgsz % 32 != 0:
        print(f"[ERREUR] imgsz doit etre un multiple de 32 : {imgsz}")
        ok = False

    if not verifier_compilateur():
        ok = False

    memoire = memoire_disponible_go()
    if memoire is None:
        print("[INFO] Memoire disponible non determinee.")
    else:
        print(f"[INFO] Memoire disponible : {memoire:.1f} Go")
        if memoire < MIN_AVAILABLE_GB:
            print(
                f"[ATTENTION] Moins de {MIN_AVAILABLE_GB:.0f} Go disponibles. "
                "Le processus risque d'etre tue (Killed). "
                "Augmente la memoire dans .wslconfig ou reduis --fraction."
            )

    return ok


def trouver_modele_edgetpu(dossier_poids: Path) -> Path | None:
    """Cherche le fichier *_edgetpu.tflite le plus recent pres des poids."""
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

    # Les fichiers d'export sont ecrits pres des poids d'origine.
    # La quantification INT8 est activee automatiquement pour edgetpu.
    try:
        exporte = model.export(
            format="edgetpu",
            imgsz=imgsz,
            data=str(DATASET_YAML),
            fraction=fraction,
            device="cpu",
        )
    except Exception as exc:
        print(f"\n[ERREUR] Echec de l'export : {exc}")
        print("Consulte le journal complet ci-dessus.")
        return 2

    print(f"\n[INFO] Chemin retourne par Ultralytics : {exporte}")

    modele = None
    if exporte:
        chemin = Path(str(exporte))
        if chemin.is_file() and chemin.name.endswith("_edgetpu.tflite"):
            modele = chemin

    if modele is None:
        modele = trouver_modele_edgetpu(BEST_PT.parent)

    if modele is None:
        print("[ERREUR] Aucun fichier *_edgetpu.tflite trouve.")
        print(
            "Un fichier INT8 non compile peut exister dans "
            f"{BEST_PT.parent}. Compile-le manuellement avec edgetpu_compiler."
        )
        return 3

    destination = OUTPUT_DIR / modele.name
    shutil.copy2(modele, destination)

    print("\n[OK] Export termine.")
    print(f" - Modele compile : {modele}")
    print(f" - Copie          : {destination.resolve()}")

    journal = modele.with_suffix(".log")
    if journal.exists():
        print(f" - Rapport compilateur : {journal}")
        shutil.copy2(journal, OUTPUT_DIR / journal.name)

    print("\nA verifier dans le rapport du compilateur :")
    print(" - nombre d'operations executees sur Edge TPU ;")
    print(" - nombre d'operations laissees sur CPU.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO vers Coral Edge TPU.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Resolution d'export (identique a celle retenue pour le projet).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=DEFAULT_FRACTION,
        help="Fraction du dataset utilisee pour la calibration INT8.",
    )
    args = parser.parse_args()
    return exporter(args.imgsz, args.fraction)


if __name__ == "__main__":
    sys.exit(main())
