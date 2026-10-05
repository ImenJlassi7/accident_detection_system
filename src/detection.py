"""
Module 3 - Détection et classification (YOLO)
Détecte les véhicules dans une frame et renvoie leurs bounding boxes,
classes et scores de confiance.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from ultralytics import YOLO

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str
    confidence: float

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def bbox_xyxy(self):
        return (self.x1, self.y1, self.x2, self.y2)


class VehicleDetector:
    """
    Enveloppe YOLO pour la détection de véhicules.

    Deux modes, pilotés par config.USE_CUSTOM_MODEL :
    - False (par défaut) : YOLOv8n pré-entraîné COCO, pour démarrer/tester
      le pipeline immédiatement, sans entraînement.
    - True : modèle fine-tuné sur le dataset complet 8219 images / 8 classes
      (auto, bus, car, lcv, motorcycle, multiaxle, tractor, truck) via
      prepare_dataset.py + train.py (à lancer sur Colab/Kaggle, GPU requis).
    """

    def __init__(self, model_path: str = None, confidence: float = None,
                 classes: dict = None):
        self.model = YOLO(model_path or config.YOLO_MODEL_PATH)
        self.confidence = confidence or config.CONFIDENCE_THRESHOLD
        self.classes = classes or config.VEHICLE_CLASSES

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=list(self.classes.keys()),
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_id=cls_id,
                    class_name=self.classes.get(cls_id, str(cls_id)),
                    confidence=float(box.conf),
                ))
        return detections


if __name__ == "__main__":
    import cv2
    import sys as _sys
    from acquisition import FrameSource

    source = _sys.argv[1] if len(_sys.argv) > 1 else str(config.SAMPLE_IMAGES_DIR)
    detector = VehicleDetector()
    src = FrameSource(source)

    total_detections = 0
    for frame_id, frame in src.frames():
        dets = detector.detect(frame)
        total_detections += len(dets)
        print(f"Frame {frame_id}: {len(dets)} véhicule(s) -> "
              f"{[d.class_name for d in dets]}")

    print(f"Total : {total_detections} détections")
