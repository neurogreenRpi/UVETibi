"""
Export complet YOLO (.pt) -> modele Coral Edge TPU, avec patch des redimensionnements.

A executer sous Linux / WSL2 x86_64, dans l'environnement coral_export.

Etapes :
 1. Verifications (poids, dataset, compilateur, memoire).
 2. Export Ultralytics en saved_model INT8 (PyTorch -> ONNX -> TF -> TFLite INT8).
    Ultralytics n'appelle PAS le compilateur (ses options ne sont pas
    acceptees par la v15).
 3. Patch : RESIZE_NEAREST_NEIGHBOR version 3 -> version 2
    (halfPixelCenters=False), avec comparaison des sorties.
 4. Compilation Edge TPU avec le compilateur candidat (options -s -o).
 5. Lecture du rapport et copie dans exports_coral/.

Usage :
    python export_coral.py
    python export_coral.py --imgsz 416 --fraction 0.1
    python export_coral.py --skip-export     # reutilise le TFLite INT8 existant
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# =====================================================
# CONFIGURATION
# =====================================================

BEST_PT = Path(
    "/mnt/a/best.pt"
)
DATASET_YAML = Path("/mnt/a/Camion/data/data_linux.yaml")

DEFAULT_IMGSZ = 416
DEFAULT_FRACTION = 0.1
OUTPUT_DIR = Path("exports_coral")
MIN_AVAILABLE_GB = 6.0
COMPILE_TIMEOUT_S = 300

COMPILER_DIR = Path("/root/tools/coral-compiler-candidate")
LOADER = COMPILER_DIR / "ld-linux-x86-64.so.2"
COMPILER = COMPILER_DIR / "edgetpu_compiler"

INT8_NAME = "best_full_integer_quant.tflite"
PATCHED_SUFFIX = "_resize_v2"

# =====================================================
# OUTILS
# =====================================================


def memoire_disponible_go():
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for ligne in f:
                if ligne.startswith("MemAvailable:"):
                    return int(ligne.split()[1]) / 1024 / 1024
    except OSError:
        return None
    return None


def commande_compilateur(*args):
    return [str(LOADER), "--library-path", str(COMPILER_DIR), str(COMPILER), *args]


def verifier_compilateur():
    for fichier in (LOADER, COMPILER):
        if not fichier.is_file():
            print(f"[ERREUR] Fichier introuvable : {fichier}")
            return False
    try:
        r = subprocess.run(
            commande_compilateur("--version"),
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[ERREUR] Impossible d'executer le compilateur : {exc}")
        return False
    sortie = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        print(f"[ERREUR] Le compilateur renvoie une erreur :\n{sortie}")
        return False
    print(f"[OK] {sortie}")
    return True


def verifier_prerequis(imgsz, fraction, skip_export):
    ok = True
    if not BEST_PT.exists():
        print(f"[ERREUR] Poids introuvables : {BEST_PT}")
        ok = False
    if not skip_export and not DATASET_YAML.exists():
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
        if memoire < MIN_AVAILABLE_GB and not skip_export:
            print(f"[ATTENTION] Moins de {MIN_AVAILABLE_GB:.0f} Go disponibles : "
                  "risque de processus tue. Augmente la memoire (.wslconfig) "
                  "ou reduis --fraction.")
    return ok


# =====================================================
# ETAPE 2 : EXPORT INT8
# =====================================================


def chemin_tflite_int8():
    return BEST_PT.parent / (BEST_PT.stem + "_saved_model") / INT8_NAME


def exporter_int8(imgsz, fraction):
    from ultralytics import YOLO

    model = YOLO(str(BEST_PT))
    try:
        model.export(
            format="saved_model",
            int8=True,
            imgsz=imgsz,
            data=str(DATASET_YAML),
            fraction=fraction,
            device="cpu",
        )
    except Exception as exc:
        print(f"\n[ERREUR] Echec de l'export : {exc}")
        return None

    tflite = chemin_tflite_int8()
    if not tflite.is_file():
        trouves = sorted(
            BEST_PT.parent.rglob(INT8_NAME),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not trouves:
            print(f"[ERREUR] {INT8_NAME} introuvable pres de {BEST_PT.parent}")
            return None
        tflite = trouves[0]
    return tflite


# =====================================================
# ETAPE 3 : PATCH DES REDIMENSIONNEMENTS
# =====================================================


def patcher_resize(source: Path, destination: Path) -> bool:
    import flatbuffers
    from ai_edge_litert.interpreter import Interpreter
    from tensorflow.lite.python import schema_py_generated as schema

    if destination.exists():
        print(f"[ERREUR] Destination deja existante, je n'ecrase pas : {destination}")
        return False

    donnees = bytearray(source.read_bytes())
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(donnees, 0))

    resize_code = schema.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR
    trouves = []
    for i_sg, sg in enumerate(model.subgraphs):
        for i_op, op in enumerate(sg.operators):
            oc = model.operatorCodes[op.opcodeIndex]
            code = max(oc.builtinCode, oc.deprecatedBuiltinCode)
            if code == resize_code:
                trouves.append((i_sg, i_op, sg, op, oc))

    if len(trouves) != 2:
        print(f"[ERREUR] {len(trouves)} redimensionnement(s) trouve(s), 2 attendus. "
              "Le patch ne s'applique pas a cette structure.")
        return False

    for i_sg, i_op, sg, op, oc in trouves:
        if oc.version != 3:
            print(f"[ERREUR] Op {i_op} : version {oc.version}, 3 attendue.")
            return False
        opts = op.builtinOptions
        if opts is None:
            print(f"[ERREUR] Op {i_op} : options absentes.")
            return False
        if opts.alignCorners or not opts.halfPixelCenters:
            print(f"[ERREUR] Op {i_op} : align_corners/half_pixel_centers inattendus.")
            return False

        t_in = sg.tensors[op.inputs[0]]
        t_out = sg.tensors[op.outputs[0]]
        if t_in.type != schema.TensorType.INT8 or t_out.type != schema.TensorType.INT8:
            print(f"[ERREUR] Op {i_op} : tenseurs non INT8.")
            return False
        a, b = list(t_in.shape), list(t_out.shape)
        if (len(a) != 4 or len(b) != 4 or a[0] != b[0] or a[3] != b[3]
                or b[1] != 2 * a[1] or b[2] != 2 * a[2]):
            print(f"[ERREUR] Op {i_op} : facteur x2 non verifie ({a} -> {b}).")
            return False
        print(f"[PATCH] Sous-graphe {i_sg}, operation {i_op} : {tuple(a)} -> {tuple(b)}")

    # Un code d'operation partage ne doit servir qu'a ces deux operations.
    codes_patches = {id(oc) for *_, oc in trouves}
    for sg in model.subgraphs:
        for op in sg.operators:
            oc = model.operatorCodes[op.opcodeIndex]
            if id(oc) in codes_patches and max(
                    oc.builtinCode, oc.deprecatedBuiltinCode) != resize_code:
                print("[ERREUR] Code d'operation partage avec une autre operation.")
                return False

    for *_, op, oc in [(t[0], t[1], t[3], t[4]) for t in trouves]:
        op.builtinOptions.halfPixelCenters = False
        oc.version = 2
    model.signatureDefs = []

    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    contenu = bytes(builder.Output())

    # Comparaison des sorties original / patche.
    it_a = Interpreter(model_content=source.read_bytes())
    it_b = Interpreter(model_content=contenu)
    it_a.allocate_tensors()
    it_b.allocate_tensors()

    in_a, in_b = it_a.get_input_details(), it_b.get_input_details()
    out_a, out_b = it_a.get_output_details(), it_b.get_output_details()
    if len(in_a) != 1 or len(in_b) != 1:
        print("[ERREUR] Une seule entree attendue.")
        return False
    if in_a[0]["dtype"] != np.int8 or in_a[0]["shape"].tolist() != in_b[0]["shape"].tolist():
        print("[ERREUR] Entree inattendue (INT8 requis, formes identiques).")
        return False
    if len(out_a) != len(out_b):
        print("[ERREUR] Nombre de sorties different.")
        return False
    for da, db in zip(out_a, out_b):
        if (da["shape"].tolist() != db["shape"].tolist()
                or da["dtype"] != db["dtype"]
                or tuple(da["quantization"]) != tuple(db["quantization"])):
            print("[ERREUR] Sortie differente entre original et patche.")
            return False

    forme = tuple(in_a[0]["shape"])
    rng = np.random.default_rng(42)
    entrees = [
        np.full(forme, -128, dtype=np.int8),
        np.zeros(forme, dtype=np.int8),
        rng.integers(-128, 128, size=forme, dtype=np.int8),
    ]
    for n, x in enumerate(entrees, 1):
        it_a.set_tensor(in_a[0]["index"], x)
        it_b.set_tensor(in_b[0]["index"], x)
        it_a.invoke()
        it_b.invoke()
        for da, db in zip(out_a, out_b):
            if not np.array_equal(it_a.get_tensor(da["index"]),
                                  it_b.get_tensor(db["index"])):
                print(f"[ERREUR] Test {n} : sorties differentes. Rien n'est ecrit.")
                return False
        print(f"[OK] Test {n} : sorties strictement identiques.")

    destination.write_bytes(contenu)
    print(f"[OK] Modele modifie : {destination} ({len(contenu)} octets)")
    print("     Controle preliminaire : valider ensuite sur images reelles et Coral.")
    return True


# =====================================================
# ETAPE 4 : COMPILATION
# =====================================================


def compiler_edgetpu(modele: Path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = commande_compilateur("-s", "-o", str(OUTPUT_DIR), str(modele))
    print("[INFO] Compilation Edge TPU...")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=COMPILE_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        print(f"[ERREUR] Compilation interrompue apres {COMPILE_TIMEOUT_S} s.")
        return None, ""
    except OSError as exc:
        print(f"[ERREUR] Compilateur inutilisable : {exc}")
        return None, ""

    sortie = r.stdout + r.stderr
    (OUTPUT_DIR / "compile.log").write_text(sortie, encoding="utf-8")
    print(sortie)
    if r.returncode != 0:
        print(f"[ERREUR] Code de sortie du compilateur : {r.returncode}")
        return None, sortie

    final = OUTPUT_DIR / (modele.stem + "_edgetpu.tflite")
    if not final.is_file():
        print(f"[ERREUR] Fichier attendu absent : {final}")
        return None, sortie
    return final, sortie


def resumer_rapport(sortie: str):
    tpu = re.search(r"operations that will run on Edge TPU:\s*(\d+)", sortie)
    cpu = re.search(r"operations that will run on CPU:\s*(\d+)", sortie)
    sg = re.search(r"Number of Edge TPU subgraphs:\s*(\d+)", sortie)
    print("\n[RESUME] Rapport du compilateur")
    print(f" - Operations sur Edge TPU : {tpu.group(1) if tpu else '?'}")
    print(f" - Operations sur CPU      : {cpu.group(1) if cpu else '?'}")
    print(f" - Sous-graphes Edge TPU   : {sg.group(1) if sg else '?'}")
    if "RESIZE_NEAREST_NEIGHBOR" in sortie:
        for ligne in sortie.splitlines():
            if "RESIZE_NEAREST_NEIGHBOR" in ligne:
                print(f" - {ligne.strip()}")


# =====================================================
# PROGRAMME PRINCIPAL
# =====================================================


def main():
    parser = argparse.ArgumentParser(description="Export YOLO .pt -> Edge TPU.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--skip-export", action="store_true",
                        help="Reutilise le TFLite INT8 deja present.")
    args = parser.parse_args()

    if not verifier_prerequis(args.imgsz, args.fraction, args.skip_export):
        return 1

    print("\n[INFO] Configuration")
    print(f" - Poids    : {BEST_PT}")
    print(f" - Dataset  : {DATASET_YAML}")
    print(f" - imgsz    : {args.imgsz}")
    print(f" - fraction : {args.fraction}")
    print(f" - Sortie   : {OUTPUT_DIR.resolve()}\n")

    if args.skip_export:
        tflite = chemin_tflite_int8()
        if not tflite.is_file():
            print(f"[ERREUR] TFLite INT8 introuvable : {tflite}")
            return 2
        print(f"[INFO] Reutilisation de {tflite}")
    else:
        tflite = exporter_int8(args.imgsz, args.fraction)
        if tflite is None:
            return 2
    print(f"[OK] TFLite INT8 : {tflite}")

    patche = tflite.with_name(tflite.stem + PATCHED_SUFFIX + ".tflite")
    if not patcher_resize(tflite, patche):
        return 3

    final, sortie = compiler_edgetpu(patche)
    if final is None:
        return 4

    resumer_rapport(sortie)
    journal = final.with_suffix(".log")
    print(f"\n[OK] Modele final : {final.resolve()}")
    if journal.exists():
        print(f"[OK] Rapport detaille : {journal.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
