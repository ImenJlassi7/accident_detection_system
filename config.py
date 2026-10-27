"""
Configuration centrale du système de détection d'accidents routiers.
Modifier ce fichier plutôt que de coder les valeurs en dur dans les modules.
"""

from pathlib import Path

# --- Chemins ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
OUTPUT_DIR = BASE_DIR / "outputs"          # rapports, frames annotées, logs
DB_PATH = BASE_DIR / "outputs" / "events.db"

# --- Module Détection (YOLO) ---
# Le dataset complet (8219 images + labels, classes.txt) EST disponible :
# dataset Kaggle "Vehicle Detection - 8 Classes". Utiliser prepare_dataset.py
# puis train.py (sur Colab/Kaggle, GPU requis) pour obtenir un modèle
# fine-tuné best.pt sur ces 8 classes précises (plus adapté au trafic
# indien/tunisien que les classes génériques COCO).
#
# USE_CUSTOM_MODEL = False -> démarrage rapide avec YOLOv8n pré-entraîné COCO
# USE_CUSTOM_MODEL = True  -> pointer YOLO_MODEL_PATH vers le best.pt obtenu
#                              après entraînement sur le dataset 8 classes
USE_CUSTOM_MODEL = False
CONFIDENCE_THRESHOLD = 0.35

if USE_CUSTOM_MODEL:
    YOLO_MODEL_PATH = "runs_accident_detection/vehicle8_finetune/weights/best.pt"
    # Classes du dataset (train/labels/classes.txt), id -> nom
    VEHICLE_CLASSES = {
        0: "auto",
        1: "bus",
        2: "car",
        3: "lcv",
        4: "motorcycle",
        5: "multiaxle",
        6: "tractor",
        7: "truck",
    }
else:
    YOLO_MODEL_PATH = "yolo26n.pt"
    # Classes COCO pertinentes pour le trafic routier (id COCO -> nom)
    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# --- Module Tracking (étape suivante) ---
TRACKER_TYPE = "bytetrack"  # ou "deepsort"

# --- Module Détection d'accident (étape suivante) ---
ACCIDENT_MIN_PERSISTENCE_FRAMES = 5   # nb frames consécutives pour confirmer
IOU_OVERLAP_THRESHOLD = 0.15          # chevauchement bounding boxes
SUDDEN_STOP_SPEED_DROP_RATIO = 0.7    # chute de vitesse considérée comme arrêt brusque

# --- Localisation caméra (étape suivante) ---
# Coordonnées GPS fixes de la caméra (à renseigner par site de déploiement)
CAMERA_GPS = {
    "latitude": None,
    "longitude": None,
}

# --- Alerte (étape suivante) ---
ALERT_EMAIL_RECIPIENTS = []
ALERT_SMS_RECIPIENTS = []
