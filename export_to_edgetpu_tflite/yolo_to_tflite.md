# Entraînement YOLOv9 et export pour Coral Edge TPU

## 1. Objectif

Ce projet consiste à :

1. Préparer un dataset de détection exporté depuis CVAT au format YOLO.
2. Entraîner un modèle YOLOv9 sous Windows.
3. Exporter et compiler le modèle pour Google Coral Edge TPU sous Linux/WSL.
4. Déployer le modèle sur un Raspberry Pi associé au Coral.
5. Vérifier la précision et les performances réelles du système.

Configuration du projet :

| Élément | Configuration |
|---|---|
| Images source | 640 × 480 pixels |
| Résolution du modèle | 416 × 416 pixels |
| Modèle de départ | `yolov9t.pt` |
| Classes | `marche_avant`, `creneau` |
| Entraînement | Windows, environnement Python existant |
| Export | WSL/Linux x86_64, Python géré par pyenv |
| Inférence | Raspberry Pi + Coral Edge TPU |

> YOLOv9 est ici un candidat à évaluer, pas une garantie d’exécution intégrale
> sur Coral. La compatibilité effective dépend du graphe exporté, de sa
> quantification et du résultat de compilation.

---

## 2. Organisation des fichiers

Organisation conseillée du projet :

```text
projet/
├── README.md
├── train_yolov9_coral.py
├── export_coral.py
├── requirements-windows.txt
├── requirements-export.txt
└── runs/
    └── detect/
        └── yolo_camions_edgetpu_416/
            └── weights/
                ├── best.pt
                └── last.pt
```

Rôle des scripts :

| Script | Rôle | Environnement |
|---|---|---|
| `train_yolov9_coral.py` | Préparation du dataset et entraînement | Windows |
| `export_coral.py` | Conversion et compilation Edge TPU | WSL/Linux |

Le chemin réel des résultats et des fichiers exportés est celui indiqué dans
les logs des scripts. Ne pas supposer que les paramètres `project` et `name`
de l’export produisent la même arborescence que ceux de l’entraînement.

---

## 3. Préparer le dataset CVAT

### 3.1 Format attendu

Exporter les annotations CVAT au format YOLO Detection.

Chaque image annotée doit avoir un fichier `.txt` de même nom :

```text
image_001.jpg
image_001.txt
```

Chaque ligne d’annotation contient :

```text
class_id x_center y_center width height
```

Les coordonnées sont normalisées entre 0 et 1.

Correspondance des classes :

```text
0 : marche_avant
1 : creneau
```

Cette correspondance doit rester identique entre CVAT, le fichier YAML,
l’entraînement et l’application d’inférence.

### 3.2 Organisation du dataset

Après préparation :

```text
A:\data\
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

Exemple de `data.yaml` pour Windows :

```yaml
path: A:/data
train: images/train
val: images/val

names:
  0: marche_avant
  1: creneau
```

### 3.3 Séparation entraînement/validation

Séparer les données par session vidéo, et non simplement par image.

Des images successives d’une même vidéo sont très similaires. Les répartir
entre entraînement et validation peut donner une estimation trop optimiste
des performances.

Vérifications :

- Aucune session ne doit apparaître dans les deux ensembles.
- Les deux classes doivent être suffisamment représentées dans chaque ensemble.
- Les conditions d’éclairage et les distances doivent être variées.
- Le jeu de validation doit contenir des images exploitables.
- Conserver si possible des sessions supplémentaires pour un test final indépendant.

> Faire une sauvegarde du dataset avant de lancer un script qui déplace
> physiquement les images et les annotations.

### 3.4 Images sans objet

Une image réellement sans objet cible peut être conservée comme background.

Attention : un fichier d’annotation absent ne prouve pas que l’image est un
background. Il peut aussi signaler une annotation manquante ou un export incomplet.

Ne créer automatiquement des annotations vides qu’après avoir vérifié ce point.

---

## 4. Entraînement sous Windows

### 4.1 Activer l’environnement existant

Utiliser l’environnement Python Windows déjà configuré pour l’entraînement.

Vérifier l’interpréteur et les bibliothèques :

```powershell
python --version
python -m pip --version

python -c "import torch; print('PyTorch :', torch.__version__); print('CUDA disponible :', torch.cuda.is_available())"

python -c "import ultralytics; print('Ultralytics :', ultralytics.__version__)"
```

Si le script utilise `device=0`, un GPU compatible doit être disponible.
Sinon, adapter le paramètre au matériel utilisé.

Ne pas remplacer l’installation PyTorch GPU de Windows par une version CPU :
l’installation CPU décrite plus loin concerne uniquement l’environnement WSL
dédié à l’export.

### 4.2 Paramètres à vérifier dans le script

Avant de lancer `train_yolov9_coral.py`, vérifier :

| Paramètre | Valeur initiale |
|---|---|
| Dataset | `A:\data` |
| Classes | `marche_avant`, `creneau` |
| Poids initiaux | `yolov9t.pt` |
| Résolution | `416` |
| Époques maximales | `120` |
| Batch | `16`, à adapter à la mémoire GPU |
| Patience | `25` |
| Nom de l’expérience | `yolo_camions_edgetpu_416` |

Ces valeurs constituent un point de départ, pas des contraintes du Coral.

Vérifier également les augmentations :

- Les retournements doivent rester cohérents avec la signification des classes.
- Les rotations doivent correspondre à des situations plausibles.
- Les augmentations de couleur et de luminosité ne doivent pas dénaturer les images.

### 4.3 Lancer l’entraînement

Depuis le dossier du projet :

```powershell
python train_yolov9_coral.py
```

Le chemin exact de sauvegarde est affiché à la fin.

Les poids attendus sont généralement :

```text
runs/detect/yolo_camions_edgetpu_416/weights/best.pt
runs/detect/yolo_camions_edgetpu_416/weights/last.pt
```

Attention : Ultralytics peut créer un dossier suffixé si une expérience du
même nom existe déjà.

### 4.4 Analyser les résultats

Examiner notamment :

- La précision et le rappel par classe.
- Le mAP50 et le mAP50–95.
- La matrice de confusion.
- Les images de validation avec prédictions.
- Les faux positifs sur les images de background.
- Les erreurs sur les camions éloignés, partiels ou peu éclairés.

Exporter de préférence `best.pt`, puis conserver les résultats de validation
comme référence avant quantification.

### 4.5 Sauvegarder l’environnement Windows

```powershell
python -m pip freeze > requirements-windows.txt
```

Conserver également le dataset utilisé, la répartition des sessions et les
paramètres d’entraînement.

---

## 5. Préparer l’environnement d’export WSL avec pyenv

### 5.1 Principe

L’environnement WSL est réservé à l’export.

- pyenv sélectionne la version de Python.
- Un environnement virtuel isole les dépendances.
- PyTorch CPU suffit pour la conversion.
- Le Coral n’a pas besoin d’être branché pour compiler le modèle.

Ne pas utiliser `sudo pip install`.

Utiliser systématiquement :

```bash
python -m pip
```

afin d’installer les paquets dans le Python actif.

### 5.2 Vérifier la configuration existante

```bash
uname -m

pyenv versions
pyenv version

python --version
python -c "import sys; print(sys.executable)"
python -m pip --version
```

Pour le compilateur Edge TPU utilisé dans cette démarche, travailler sur une
distribution Linux/WSL x86_64.

Le Python et pip doivent correspondre à l’environnement d’export, pas au
Python système de WSL.

### 5.3 Cas A : environnement géré par pyenv-virtualenv

Si `coral_export` apparaît dans les environnements pyenv :

```bash
pyenv activate coral_export
```

Pour l’associer au dossier du projet :

```bash
cd ~/ton-projet
pyenv local coral_export
```

Les commandes `pyenv activate` et `pyenv virtualenv` nécessitent le plugin
`pyenv-virtualenv`.

Si l’environnement est déjà correctement activé, ne pas le recréer.

### 5.4 Cas B : pyenv associé à un environnement venv classique

Si pyenv gère uniquement la version de Python, sélectionner une version déjà
installée :

```bash
cd ~/ton-projet
pyenv versions
```

Puis remplacer `VERSION_INSTALLEE` par son identifiant exact :

```bash
pyenv local VERSION_INSTALLEE
python -m venv .venv
source .venv/bin/activate
```

Si un environnement virtuel existe déjà, l’activer au lieu d’en créer un autre.

### 5.5 Installer les dépendances

Actualiser les outils d’installation :

```bash
python -m pip install --upgrade pip setuptools wheel
```

Installer PyTorch et torchvision CPU :

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --timeout 120 --retries 10 --resume-retries 20
```

Installer ensuite Ultralytics :

```bash
python -m pip install ultralytics --timeout 120 --retries 10 --resume-retries 20
```

Si une installation GPU a déjà été effectuée par erreur, retirer d’abord les
paquets concernés dans cet environnement d’export uniquement :

```bash
python -m pip uninstall -y torch torchvision torchaudio triton
```

Puis reprendre l’installation CPU.

### 5.6 Vérifier l’installation

```bash
python -c "import sys; print(sys.executable)"

python -c "import torch; print('PyTorch :', torch.__version__); print('CUDA :', torch.version.cuda)"

python -c "import ultralytics; print('Ultralytics :', ultralytics.__version__)"


python -m pip check
```

Pour l’environnement CPU, `torch.version.cuda` doit normalement afficher `None`.

Les dépendances de conversion supplémentaires dépendent de la version
d’Ultralytics. Examiner les messages de l’export et les contraintes indiquées
avant d’installer manuellement des versions de TensorFlow, ONNX ou onnx2tf.

> Ne pas supposer qu’installer arbitrairement les dernières versions de tous
> les convertisseurs produit un ensemble compatible.

---

## 6. Préparer les chemins Linux et la calibration

### 6.1 Accès aux fichiers Windows

Exemples de correspondance :

| Windows | WSL |
|---|---|
| `C:\projet\best.pt` | `/mnt/c/projet/best.pt` |
| `A:\data` | `/mnt/a/data`, si le lecteur est monté |

Vérifier les chemins :

```bash
ls /mnt/a/data
ls /mnt/c/projet/best.pt
```

Un lecteur Windows n’est pas nécessairement monté automatiquement dans tous les
cas. Adapter les chemins à la configuration réelle de WSL.

### 6.2 Créer un YAML spécifique à Linux

Ne pas utiliser tel quel un YAML contenant :

```yaml
path: A:/data
```

Créer par exemple `data_linux.yaml` :

```yaml
path: /mnt/a/data
train: images/train
val: images/val

names:
  0: marche_avant
  1: creneau
```

Le script d’export doit pointer vers cette version Linux.

### 6.3 Calibration INT8

La quantification doit utiliser des images représentatives de l’application :

- Images avec camions.
- Images sans camion.
- Différentes distances.
- Occultations partielles.
- Conditions d’éclairage réellement rencontrées.

Vérifier dans les logs le split et le nombre d’images effectivement utilisés
pour la calibration.

Conserver si possible un jeu de test indépendant de la calibration pour
évaluer la précision après conversion.

---

## 7. Export et compilation Edge TPU

### 7.1 Vérifier le script d’export

Dans `export_coral.py`, contrôler :

| Élément | Valeur attendue |
|---|---|
| Poids source | Chemin Linux vers le bon `best.pt` |
| Dataset | Chemin vers `data_linux.yaml` |
| Format | `edgetpu` |
| Résolution | `416` |
| Périphérique de conversion | `cpu` |
| Batch d’export | `1` |
| Formes dynamiques | Désactivées |
| NMS intégré | Désactivé pour cette démarche |

Ne pas utiliser le paramètre `quantize=8` indiqué dans une ancienne version
du script : ce n’est pas le paramètre adapté à cette procédure.

L’export Edge TPU nécessite une quantification entière compatible avec le
compilateur Coral.

### 7.2 Lancer l’export

Dans le bon environnement pyenv/virtuel :

```bash
python export_coral.py
```

Pour conserver les logs :

```bash
set -o pipefail
python export_coral.py 2>&1 | tee export_coral.log
```

L’export peut télécharger des dépendances et prendre plusieurs minutes.

### 7.3 Fichier final

Le modèle destiné au Coral se termine normalement par :

```text
_edgetpu.tflite
```

Exemple :

```text
best_full_integer_quant_edgetpu.tflite
```

Utiliser le chemin réellement retourné par le script.

Un fichier `.tflite` quantifié non compilé n’est pas équivalent au fichier
compilé pour Edge TPU.

### 7.4 Compilateur système

`edgetpu_compiler` est un programme Linux, pas un paquet Python géré par pyenv.

Vérifier sa disponibilité :

```bash
command -v edgetpu_compiler
edgetpu_compiler --version
```

Selon la version d’Ultralytics, l’export peut tenter de l’installer
automatiquement. Si cette étape échoue, traiter séparément l’installation du
compilateur système.

Pour compiler manuellement le modèle INT8 intermédiaire :

```bash
mkdir -p compiled

edgetpu_compiler -s \
    -o compiled \
    /chemin/vers/best_full_integer_quant.tflite
```

---

## 8. Interpréter le rapport de compilation

Examiner :

- Les opérations exécutées sur Edge TPU.
- Les opérations laissées au CPU.
- La nature et la position des opérations non mappées.
- Les éventuelles limites de mémoire et l’utilisation du cache des paramètres.

Ne pas juger le modèle uniquement au nombre d’opérations CPU.

Une seule opération coûteuse ou située tôt dans le réseau peut être plus
pénalisante que plusieurs opérations légères en fin de graphe.

De même, un modèle entièrement mappé sur TPU n’exonère pas le Raspberry Pi :

- Du décodage vidéo.
- Du redimensionnement et du letterboxing.
- Du traitement des sorties.
- Du NMS.
- De la logique applicative.

La décision finale doit reposer sur la latence réelle et la précision du
pipeline complet.

---

## 9. Déploiement sur Raspberry Pi

### 9.1 Copier les éléments utiles

Copier :

- Le modèle `_edgetpu.tflite`.
- La correspondance des classes si nécessaire.
- Le code d’inférence adapté à cet export.
- Les paramètres de prétraitement et de post-traitement.

Exemple :

```bash
scp /chemin/vers/modele_edgetpu.tflite \
    utilisateur@IP_DU_PI:/home/utilisateur/models/
```

Adapter l’utilisateur, l’adresse IP et le dossier de destination.

### 9.2 Environnement d’inférence

L’environnement du Raspberry Pi est distinct de l’environnement WSL.

Il doit disposer :

- Du runtime Coral `libedgetpu`.
- D’un interpréteur TFLite compatible avec ce runtime.
- Du code permettant de charger le délégué Edge TPU.
- Des dépendances de l’application vidéo.

Ne pas supposer que les versions Python de WSL et du Raspberry Pi doivent être
identiques : leurs contraintes de dépendances peuvent différer.

Ne pas supposer non plus que `device="tpu"` est un argument universel :
le chargement du Coral dépend du backend d’inférence utilisé.

### 9.3 Prétraitement

Pour une image caméra de 640 × 480 vers une entrée carrée de 416 × 416 :

- Conserver le ratio d’aspect.
- Redimensionner le contenu vers environ 416 × 312.
- Ajouter le padding nécessaire pour atteindre 416 × 416.
- Appliquer l’ordre des canaux et la quantification attendus par le modèle.
- Inverser correctement le letterboxing pour retrouver les coordonnées source.

Éviter d’étirer directement l’image 640 × 480 en carré si ce traitement n’est
pas celui utilisé pour l’export et la validation.

---

## 10. Validation finale

Comparer les performances avant et après conversion.

| Mesure | PyTorch | TFLite INT8 / Edge TPU |
|---|---|---|
| Précision par classe | À relever | À relever |
| Rappel par classe | À relever | À relever |
| mAP50 | À relever | À relever si l’outillage le permet |
| mAP50–95 | À relever | À relever si l’outillage le permet |
| Faux positifs | À relever | À relever |
| Latence d’inférence | Référence | À mesurer sur le Pi |
| Latence totale | Selon application | À mesurer sur le Pi |

Pour les mesures sur Raspberry Pi :

- Effectuer plusieurs inférences d’échauffement.
- Mesurer séparément l’inférence et le pipeline complet.
- Utiliser les mêmes images et les mêmes seuils de comparaison.
- Observer la température et la stabilité sur une durée représentative.

La bonne métrique n’est pas uniquement le FPS maximal : vérifier également
le délai entre la capture et la décision finale.

---

## 11. Dépannage

### Téléchargement de triton interrompu

Dans un environnement réservé à l’export CPU, vérifier d’abord si une version
GPU de PyTorch a été sélectionnée.

Installer explicitement PyTorch depuis son dépôt CPU, puis Ultralytics.

Les options de timeout et de reprise peuvent aider en cas de réseau instable,
mais ne corrigent pas une incompatibilité de dépendances.

### Mauvais Python utilisé

Vérifier :

```bash
pyenv version
python -c "import sys; print(sys.executable)"
python -m pip --version
```

Réactiver le bon environnement.

Ne pas utiliser `sudo pip`.

### Dataset introuvable sous WSL

Vérifier :

- Le montage du lecteur Windows.
- Le chemin du YAML transmis au script.
- Le champ `path` à l’intérieur du YAML.
- La présence des dossiers `images/train` et `images/val`.

### Trop d’opérations sur CPU

Examiner le rapport détaillé du compilateur.

Avant de réentraîner pendant plusieurs heures, tester l’export d’un modèle
candidat avec la résolution et les paramètres envisagés.

Changer vers un modèle plus gros ne corrige pas automatiquement les opérations
incompatibles.

### Baisse de précision après quantification

Contrôler :

- Les images réellement utilisées pour la calibration.
- Le prétraitement.
- L’ordre des canaux.
- Les paramètres de quantification des entrées et sorties.
- Le décodage des boîtes.
- Le letterboxing et son inversion.
- Les seuils de confiance et de NMS.

### Mémoire GPU insuffisante pendant l’entraînement

Réduire d’abord le batch sous Windows.

Le batch d’entraînement ne doit pas être confondu avec le batch d’export
pour le Coral.

---

## 12. Reproductibilité

Après un export fonctionnel :

```bash
python -m pip freeze > requirements-export.txt

python --version > export-environment.txt
pyenv version >> export-environment.txt
uname -m >> export-environment.txt
edgetpu_compiler --version >> export-environment.txt 2>&1
```

Conserver ensemble :

- `best.pt`.
- Le fichier `_edgetpu.tflite`.
- Les scripts utilisés.
- Les YAML Windows et Linux.
- Les versions des dépendances.
- Les logs de compilation.
- La liste des sessions train/validation/test.
- Les résultats de précision et de latence.

---

## 13. Checklist

### Avant entraînement

- [ ] Sauvegarde du dataset réalisée.
- [ ] Classes vérifiées.
- [ ] Annotations absentes distinguées des vrais backgrounds.
- [ ] Sessions séparées entre train et validation.
- [ ] Environnement Windows vérifié.
- [ ] Résolution initiale fixée à 416.
- [ ] Augmentations compatibles avec les classes.

### Avant export

- [ ] Bon `best.pt` sélectionné.
- [ ] Environnement pyenv/virtuel activé.
- [ ] PyTorch CPU disponible sous WSL.
- [ ] YAML Linux créé.
- [ ] Chemins accessibles.
- [ ] Dataset de calibration représentatif.
- [ ] Paramètres d’export vérifiés.

### Avant déploiement

- [ ] Compilation Edge TPU terminée.
- [ ] Rapport détaillé examiné.
- [ ] Fichier `_edgetpu.tflite` identifié.
- [ ] Runtime Coral installé sur le Pi.
- [ ] Prétraitement et post-traitement vérifiés.
- [ ] Précision après quantification évaluée.
- [ ] Latence totale et stabilité mesurées.
