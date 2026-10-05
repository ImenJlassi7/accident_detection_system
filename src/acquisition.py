"""
Module 1 - Acquisition
Lit soit un fichier vidéo, soit un dossier d'images (cas de notre dataset
actuel, qui ne contient que des images extraites, pas de vidéo), et fournit
un flux uniforme de frames à la suite du pipeline.
"""

from pathlib import Path
from typing import Iterator, Tuple
import cv2
import numpy as np


class FrameSource:
    """Source de frames unifiée : vidéo (.mp4, .avi...) ou dossier d'images."""

    def __init__(self, source_path: str):
        self.source_path = Path(source_path)
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source introuvable : {source_path}")
        self.is_video = self.source_path.is_file() and self.source_path.suffix.lower() in {
            ".mp4", ".avi", ".mov", ".mkv"
        }

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Génère (frame_id, frame_bgr) dans l'ordre."""
        if self.is_video:
            yield from self._frames_from_video()
        else:
            yield from self._frames_from_folder()

    def _frames_from_video(self) -> Iterator[Tuple[int, np.ndarray]]:
        cap = cv2.VideoCapture(str(self.source_path))
        frame_id = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame_id, frame
                frame_id += 1
        finally:
            cap.release()

    def _frames_from_folder(self) -> Iterator[Tuple[int, np.ndarray]]:
        image_paths = sorted(
            p for p in self.source_path.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for frame_id, path in enumerate(image_paths):
            frame = cv2.imread(str(path))
            if frame is not None:
                yield frame_id, frame


def preprocess(frame: np.ndarray, target_size: Tuple[int, int] = None,
               roi: Tuple[int, int, int, int] = None) -> np.ndarray:
    """
    Prétraitement (module 2 du cahier des charges) :
    - découpe optionnelle sur une zone d'intérêt (ROI) x1,y1,x2,y2
    - redimensionnement optionnel
    YOLO gère déjà le redimensionnement en interne ; cette fonction sert
    surtout à isoler la zone utile de la route (ex: retirer le ciel, les
    panneaux, la structure du pont visible sur les caméras de hauteur).
    """
    out = frame
    if roi is not None:
        x1, y1, x2, y2 = roi
        out = out[y1:y2, x1:x2]
    if target_size is not None:
        out = cv2.resize(out, target_size)
    return out


if __name__ == "__main__":
    import sys
    src = FrameSource(sys.argv[1] if len(sys.argv) > 1 else "data/sample_images")
    count = 0
    for frame_id, frame in src.frames():
        count += 1
    print(f"{count} frames lues depuis {src.source_path}")
