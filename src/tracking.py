"""
Module 4 - Tracking
Attribue un ID persistant à chaque véhicule détecté, image après image,
via le tracker intégré à Ultralytics (ByteTrack par défaut).
C'est ce qui permettra ensuite (module 5) de calculer vitesses et
trajectoires : sans ID stable, impossible de savoir que la "voiture vue
frame 10" est la même qu'à la frame 11.
"""

from dataclasses import dataclass
from typing import Dict, List, Iterator, Tuple
from collections import defaultdict
import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from detection import VehicleDetector, Detection


@dataclass
class TrackedDetection(Detection):
    track_id: int = -1


class VehicleTracker:
    """
    Suivi multi-objets des véhicules sur une séquence vidéo.
    Combine détection + association d'identité en un seul appel
    (model.track), via ByteTrack ou BoT-SORT (config.TRACKER_TYPE).
    """

    def __init__(self, model_path: str = None, confidence: float = None,
                 classes: dict = None, tracker: str = None):
        self.detector = VehicleDetector(model_path, confidence, classes)
        tracker_type = tracker or config.TRACKER_TYPE
        self.tracker_cfg = "bytetrack.yaml" if tracker_type == "bytetrack" else "botsort.yaml"
        # historique des positions par ID -> utilisé par le module Analyse
        # (module 5 : vitesses, trajectoires, arrêts brusques)
        self.history: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

    def track_video(self, source: str) -> Iterator[Tuple[int, List[TrackedDetection]]]:
        """Génère (frame_id, [TrackedDetection, ...]) pour chaque frame."""
        results = self.detector.model.track(
            source=source,
            conf=self.detector.confidence,
            classes=list(self.detector.classes.keys()),
            tracker=self.tracker_cfg,
            persist=True,
            stream=True,
            verbose=False,
        )
        for frame_id, r in enumerate(results):
            frame_dets = []
            if r.boxes is not None and r.boxes.id is not None:
                for box, tid in zip(r.boxes, r.boxes.id):
                    cls_id = int(box.cls)
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                    tid = int(tid)
                    det = TrackedDetection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        class_id=cls_id,
                        class_name=self.detector.classes.get(cls_id, str(cls_id)),
                        confidence=float(box.conf),
                        track_id=tid,
                    )
                    frame_dets.append(det)
                    cx, cy = det.center
                    self.history[tid].append((frame_id, cx, cy))
            yield frame_id, frame_dets

    def trajectory(self, track_id: int) -> List[Tuple[int, float, float]]:
        """Retourne l'historique (frame_id, cx, cy) d'un véhicule suivi."""
        return self.history.get(track_id, [])


def annotate_and_save(source: str, output_path: str, tracker: "VehicleTracker" = None):
    """Génère une vidéo annotée (boxes + ID) pour vérification visuelle."""
    tracker = tracker or VehicleTracker()
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    cap = cv2.VideoCapture(source)
    frame_iter = tracker.track_video(source)
    n_written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        try:
            _, dets = next(frame_iter)
        except StopIteration:
            dets = []
        for d in dets:
            x1, y1, x2, y2 = map(int, d.bbox_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID{d.track_id} {d.class_name}"
            cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        writer.write(frame)
        n_written += 1
    writer.release()
    cap.release()
    print(f"Vidéo annotée sauvegardée : {output_path} ({n_written} frames)")


if __name__ == "__main__":
    import sys as _sys
    source = _sys.argv[1] if len(_sys.argv) > 1 else str(config.DATA_DIR / "test_traffic.mp4")
    tracker = VehicleTracker()

    unique_ids = set()
    total_frames = 0
    for frame_id, dets in tracker.track_video(source):
        total_frames += 1
        for d in dets:
            unique_ids.add(d.track_id)
        print(f"Frame {frame_id}: {len(dets)} véhicule(s) suivis -> "
              f"{[(d.track_id, d.class_name) for d in dets]}")

    print(f"\n{total_frames} frames traitées, {len(unique_ids)} véhicule(s) unique(s) suivi(s)")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(config.OUTPUT_DIR / "tracked_output.mp4")
    annotate_and_save(source, out_path, tracker=VehicleTracker())
