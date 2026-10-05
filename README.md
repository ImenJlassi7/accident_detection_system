# Système de Détection d'Accidents Routiers et Alerte

Implémentation du cahier des charges (ENIS, année universitaire 2026/2027).

## Dataset (archive.zip)

8219 images + 8219 labels YOLO (`train/images`, `train/labels`), classes
définies dans `train/labels/classes.txt` :
`auto, bus, car, lcv, motorcycle, multiaxle, tractor, truck` — dataset
Kaggle "Vehicle Detection – 8 Classes".

Pas de `data.yaml` ni de split train/val fournis à la base → générés par
`prepare_dataset.py` (85% train / 15% val, seed fixe pour reproductibilité).

**Entraînement fine-tuné** : testé (pipeline validé, loss qui descend sur un
échantillon), mais **pas exécutable en pleine échelle ici** (sandbox = 1 CPU,
pas de GPU). A lancer sur Google Colab ou Kaggle comme prévu section 8 du
cahier des charges — voir "Utilisation" ci-dessous. En attendant, le projet
tourne avec YOLOv8n pré-entraîné COCO (`config.USE_CUSTOM_MODEL = False`).

## Feuille de route (mappée sur le cahier des charges)

| # | Module (cahier des charges) | Statut | Fichier |
|---|---|---|---|
| 1 | Acquisition (vidéo/frames) | ✅ Fait | `src/acquisition.py` |
| 2 | Prétraitement (ROI, resize) | ✅ Fait (fonction `preprocess`) | `src/acquisition.py` |
| 3 | Détection/classification YOLO | ✅ Fait, testé | `src/detection.py` |
| 4 | Tracking (ByteTrack/DeepSORT) | ⏳ Prochaine étape | `src/tracking.py` |
| 5 | Analyse distances/trajectoires/vitesse | ⏳ | `src/analysis.py` |
| 6 | Détection d'accident (règles) | ⏳ | `src/accident_logic.py` |
| 7 | Localisation (GPS caméra) | ⏳ | `src/geolocation.py` |
| 8 | Hôpitaux les plus proches | ⏳ | `src/geolocation.py` |
| 9 | Génération de rapport | ⏳ | `src/report.py` |
| 10 | Alerte e-mail/SMS | ⏳ | `src/alert.py` |
| 11 | Tableau de bord | ⏳ | `src/dashboard.py` (Streamlit) |

## Utilisation actuelle

```bash
pip install -r requirements.txt

# Détection avec YOLOv8n COCO (rapide, aucun entraînement requis)
python src/detection.py data/sample_images

# Préparer le dataset complet pour l'entraînement (à faire une fois)
python prepare_dataset.py /chemin/vers/archive_extraite dataset_yolo

# Entraînement fine-tuné — LANCER SUR COLAB/KAGGLE (GPU), pas ici
python train.py --data dataset_yolo/data.yaml --epochs 50
# puis dans config.py : USE_CUSTOM_MODEL = True
```
