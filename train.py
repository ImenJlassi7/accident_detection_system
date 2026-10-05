"""
Fine-tuning YOLOv8 sur le dataset "Vehicle Detection - 8 Classes".

⚠️ A lancer sur Google Colab ou Kaggle (GPU), comme prévu section 8 du
cahier des charges. Cette sandbox ne dispose que d'1 CPU sans GPU : un
entraînement complet (8218 images) y serait beaucoup trop lent.

Usage (Colab/Kaggle, après avoir exécuté prepare_dataset.py) :
    python train.py --data dataset_yolo/data.yaml --epochs 50
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset_yolo/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="checkpoint de départ (n=nano, s=small, m=medium...)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="vehicle8_finetune")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project="runs_accident_detection",
    )
    # Le meilleur poids sera dans runs_accident_detection/<name>/weights/best.pt
    # -> copier ce fichier et pointer config.YOLO_MODEL_PATH dessus.


if __name__ == "__main__":
    main()
