# Guide : du poids `best.pt` au modèle `*_edgetpu.tflite`

Ce document décrit la démarche, étape par étape, pour obtenir un modèle YOLOv9t exécutable sur Coral Edge TPU. Il suppose que l'environnement Python et le compilateur Edge TPU fonctionnent déjà (versions de référence : Ultralytics 8.4.202, TensorFlow 2.19.0, onnx2tf 1.28.8, Edge TPU Compiler 15.0.340273435).

Les résultats chiffrés cités viennent de ce projet (entrée 416x416, 2 classes : `marche_avant`, `creneau`). Ils servent de repère, pas de garantie pour un autre modèle.

## Vue d'ensemble

```
best.pt  ->  [1] contrôle  ->  [2] export INT8 (.tflite)  ->  [3] vérification
        ->  [4] patch des redimensionnements  ->  [5] compilation Edge TPU
        ->  [6] lecture du rapport  ->  [7] livrables
```

| Étape | Entrée | Sortie | Objectif |
|---|---|---|---|
| 1 | best.pt, data.yaml | rien | S'assurer que l'on part des bons poids, des bonnes classes |
| 2 | best.pt | best_full_integer_quant.tflite | Convertir en TFLite entièrement quantifié INT8 |
| 3 | .tflite INT8 | rien | Vérifier entrée/sortie avant d'aller plus loin |
| 4 | .tflite INT8 | ..._resize_v2.tflite | Rendre les redimensionnements acceptables pour le compilateur |
| 5 | ..._resize_v2.tflite | ..._edgetpu.tflite | Générer le modèle exécutable sur le Coral |
| 6 | rapport de compilation | rien | Confirmer que l'essentiel du calcul va sur le TPU |
| 7 | tous | dossier de livraison | Conserver ce qu'il faut pour reproduire et tester |

---

## Étape 0 : variables

À adapter à ton chemin. Le double `runs/detect` est celui de ce projet.

```bash
cd /mnt/a/Camion
export PT="/mnt/a/Camion/runs/detect/runs/detect/yolo_camions_edgetpu_416/weights/best.pt"
export DATA="/mnt/a/Camion/data/data_linux.yaml"
export DIR="/root/tools/coral-compiler-candidate"
export IMGSZ=416
```

`data_linux.yaml` est la version du fichier de dataset avec des chemins Linux (`/mnt/a/...`). Le `data.yaml` Windows contient des chemins `A:\...` illisibles sous WSL.

## Étape 1 : contrôler les entrées

**Objectif :** ne pas exporter un mauvais poids ou un dataset mal configuré.

À vérifier :
- `best.pt` est le poids voulu (`best`, pas `last`).
- `data_linux.yaml` liste les classes dans l'ordre de l'entraînement (ici 0 = `marche_avant`, 1 = `creneau`).
- Les chemins `val` du YAML existent sous WSL.
- La résolution d'export est celle que tu veux utiliser en production (416).

## Étape 2 : export TFLite INT8

**Objectif :** transformer le modèle PyTorch en TFLite dont les poids ET les activations sont en entiers 8 bits, avec entrée et sortie INT8. Le Coral n'exécute que du INT8.

Ce qui se passe en interne : PyTorch -> ONNX -> TensorFlow SavedModel (via onnx2tf) -> TFLite quantifié. La quantification a besoin d'images de calibration, qui sont lues depuis le split `val` du YAML. Elles doivent ressembler aux images réelles de la caméra : une calibration non représentative dégrade la précision.

```python
# export_int8.py
from ultralytics import YOLO
import os

model = YOLO(os.environ["PT"])
model.export(
    format="saved_model",
    int8=True,
    imgsz=416,
    data=os.environ["DATA"],
    fraction=0.1,     # part des images de calibration utilisée
    batch=1,
)
```

```bash
python export_int8.py 2>&1 | tee export_int8.log
```

Durée de référence : environ 8 minutes sur CPU (la conversion TensorFlow domine).

Résultat attendu, dans le dossier `best_saved_model/` à côté de `best.pt` :

```
best_full_integer_quant.tflite
```

C'est ce fichier qui est la source des étapes suivantes.

Remarques :
- Il n'est pas nécessaire d'utiliser `format="edgetpu"`. Dans ce projet, Ultralytics appelait le compilateur `/usr/local/bin/edgetpu_compiler` avec des options non reconnues par la version 14.1 (`--searchdelegate`, `--delegatesearchstep`, `--timeoutsec`), ce qui faisait échouer la phase de compilation. La conversion TFLite, elle, avait réussi. On sépare donc l'export et la compilation pour garder le contrôle sur le compilateur et ses options.
- Les messages TensorFlow sur CUDA absente ou sur des « factories » déjà enregistrées sont bénins ici : l'export tourne sur CPU.
- `fraction` et le nombre d'images de calibration : si la précision INT8 est insuffisante, augmenter `fraction` est la première piste, avant de toucher au modèle.

## Étape 3 : vérifier le TFLite INT8

**Objectif :** détecter tôt une erreur de format (type d'entrée, forme, quantification) qui rendrait inutile la suite.

```bash
python - <<'PY'
import os, glob
from ai_edge_litert.interpreter import Interpreter
p = glob.glob(os.path.dirname(os.environ["PT"]) + "/best_saved_model/best_full_integer_quant.tflite")[0]
it = Interpreter(model_path=p); it.allocate_tensors()
for d in it.get_input_details():  print("ENTREE", d["shape"], d["dtype"].__name__, d["quantization"])
for d in it.get_output_details(): print("SORTIE", d["shape"], d["dtype"].__name__, d["quantization"])
PY
```

Attendu : entrée `[1, 416, 416, 3]` en `int8`, sortie en `int8` de forme `[1, 6, 3549]` (4 valeurs de boîte + 2 classes, sur 3549 positions pour 416x416). Noter les valeurs `scale` et `zero_point` : le programme d'inférence en aura besoin.

## Étape 4 : patcher les redimensionnements

**Objectif :** supprimer la cause qui bloquait la compilation, sans réentraîner.

Le modèle contient deux opérations `RESIZE_NEAREST_NEIGHBOR` (agrandissements x2 du réseau de neurones) exportées en version 3 avec `half_pixel_centers=True`. Le compilateur Edge TPU les refuse (« Operation version not supported ») et, comme il ne gère qu'un seul sous-graphe, tout ce qui suit passe sur le CPU. Le script `patch_resize_coral.py` les convertit en version 2 avec `halfPixelCenters=False`. Pour un agrandissement entier x2 avec `align_corners=False`, les indices de pixels sont identiques, donc le résultat du modèle ne change pas.

```bash
python patch_resize_coral.py
```

Ce que fait le script :
1. Refuse d'écraser une copie existante.
2. Vérifie qu'il y a exactement 2 redimensionnements, version 3, facteur x2, INT8.
3. Écrit une copie `best_full_integer_quant_resize_v2.tflite` (l'original n'est pas modifié).
4. Compare les sorties de l'original et de la copie sur 3 entrées de test, qui doivent être strictement identiques. La copie n'est écrite que si les 3 comparaisons passent.

Si le script s'arrête sur « nombre de resize différent de 2 » ou « facteur différent de x2 », le modèle n'a pas la même structure (autre résolution, autre architecture) : le patch ne s'applique pas tel quel.

Ce contrôle est préliminaire : il ne remplace pas un test sur de vraies images.

## Étape 5 : compiler pour Edge TPU

**Objectif :** produire le fichier qui contient l'opération Edge TPU exécutée par le Coral.

```bash
SRC="$(dirname "$PT")/best_saved_model/best_full_integer_quant_resize_v2.tflite"
mkdir -p exports_coral

timeout --kill-after=10s 300s \
  "$DIR/ld-linux-x86-64.so.2" --library-path "$DIR" \
  "$DIR/edgetpu_compiler" -s -o exports_coral "$SRC" \
  2>&1 | tee compile.log
echo "code de sortie : ${PIPESTATUS[0]}"
```

Options :
- `-s` affiche le détail des opérations déléguées ou non.
- `-o` fixe le dossier de sortie.
- `timeout` évite qu'une compilation bloquée tourne indéfiniment.
- Ne pas ajouter `-a` (sous-graphes multiples, expérimental) : il n'est pas nécessaire ici.

Sortie : `exports_coral/best_full_integer_quant_resize_v2_edgetpu.tflite` et un fichier `.log` de détail.

## Étape 6 : lire le rapport

**Objectif :** savoir si le modèle exploite réellement le Coral.

Valeurs obtenues dans ce projet :

| Indicateur | Sans patch | Avec patch |
|---|---:|---:|
| Opérations sur TPU | 277 / 673 | 642 / 673 |
| Opérations sur CPU | 396 | 31 |
| Convolutions sur TPU | 77 | 185 (sur 186) |
| Sous-graphes Edge TPU | 1 | 1 |
| Paramètres diffusés hors puce | 0 | 0 |

À contrôler :
- `Model compiled successfully` et code de sortie 0.
- `RESIZE_NEAREST_NEIGHBOR ... Mapped to Edge TPU`.
- Une seule opération Edge TPU, et `Off-chip memory used for streaming ... 0.00B`.
- Les opérations restantes sur CPU : transpositions, reshapes, Softmax du décodage DFL, quelques opérations de fin de graphe. Elles sont attendues.

Le nombre d'opérations n'est pas un temps d'exécution : seule la mesure sur le Raspberry Pi donne la vitesse réelle.

## Étape 7 : livrables à conserver

```
best.pt                                      # poids d'origine
data_linux.yaml                              # classes et ordre
best_full_integer_quant.tflite               # INT8, référence CPU
best_full_integer_quant_resize_v2.tflite     # INT8 patché
best_full_integer_quant_resize_v2_edgetpu.tflite   # MODÈLE FINAL
compile.log + fichier .log du compilateur    # preuve de compilation
```

Fichier à copier sur le Raspberry Pi : `..._edgetpu.tflite`.

## Validation avant usage

La démarche s'arrête à un fichier qui compile ; elle ne prouve pas qu'il détecte bien. Avant de le déclarer bon :
- Comparer, sur des images réelles, la sortie CPU (`best_full_integer_quant.tflite`) et la sortie Coral.
- Vérifier le décodage des sorties (quantification, boîtes, NMS) dans le programme d'inférence.
- Mesurer la latence sur le Raspberry Pi avec le Coral.

## Problèmes courants

| Symptôme | Cause probable | Action |
|---|---|---|
| `Invalid option --searchdelegate` pendant l'export | Ultralytics appelle le compilateur avec des options inconnues | Exporter en `saved_model` puis compiler à la main (étapes 2 et 5) |
| `Operation version not supported` sur RESIZE | Redimensionnements en version 3 | Appliquer l'étape 4 |
| `More than one subgraph is not supported` sur presque tout | Coupure en amont (souvent un resize) | Regarder la première opération non supportée du rapport |
| Beaucoup d'opérations CPU après patch | Autre structure de modèle | Relancer l'analyse du graphe avant de modifier quoi que ce soit |
| Script Python : `Non-UTF-8 code` | Fichier enregistré en Windows-1252 | Reconvertir en UTF-8 (`iconv -f WINDOWS-1252 -t UTF-8`) |
| Accents cassés dans les logs | Console non UTF-8 | Sans conséquence sur le modèle |
