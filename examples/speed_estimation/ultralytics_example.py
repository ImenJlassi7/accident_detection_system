
from collections import defaultdict, deque

import cv2
import numpy as np
from trackers import ByteTrackTracker
from ultralytics import YOLO
import supervision as sv



# CONFIGURATION
SOURCE = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1]
])


MODEL_NAME = "yolo11n.pt"

# Résolution utilisée par YOLO
IMAGE_SIZE = 640



# TRANSFORMATION PERSPECTIVE
class ViewTransformer:

    def __init__(self, source, target):

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points):

        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)

        transformed_points = cv2.perspectiveTransform(
            reshaped_points, self.m
        )

        return transformed_points.reshape(-1, 2)


# CALCUL DE LA VITESSE
def calculate_speed(distance, elapsed_frames, fps):

    if elapsed_frames < 1:
        return 0

    elapsed_time = elapsed_frames / fps

    speed = distance / elapsed_time * 3.6

    return speed



# PROGRAMME PRINCIPAL
def main(
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
):

    print("Chargement de la vidéo...")

    video_info = sv.VideoInfo.from_video_path(
        video_path=source_video_path
    )

    print("Résolution :", video_info.resolution_wh)
    print("FPS :", video_info.fps)

    # Chargement du modèle léger
    print("Chargement du modèle YOLO11n...")

    model = YOLO(MODEL_NAME)

    # ByteTrack
    byte_track = ByteTrackTracker(
        frame_rate=video_info.fps,
        track_activation_threshold=confidence_threshold
    )

    # Annotations
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )

    text_scale = sv.calculate_optimal_text_scale(
        resolution_wh=video_info.resolution_wh
    )

    box_annotator = sv.BoxAnnotator(
        thickness=thickness
    )

    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER
    )

    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(video_info.fps * 2),
        position=sv.Position.BOTTOM_CENTER
    )

    # Zone de détection
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path
    )

    polygon_zone = sv.PolygonZone(
        polygon=SOURCE
    )

    view_transformer = ViewTransformer(
        source=SOURCE,
        target=TARGET
    )

    # Historique des positions
    coordinates = defaultdict(
        lambda: deque(maxlen=int(video_info.fps))
    )

    # Enregistrement vidéo
    with sv.VideoSink(
        target_video_path,
        video_info
    ) as sink:

        window = sv.ImageWindow("Vehicle Speed Estimation")

        print("Début du traitement...")

        for frame_index, frame in enumerate(frame_generator):

            # Détection YOLO avec résolution réduite
            result = model.predict(
                frame,
                imgsz=IMAGE_SIZE,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=True
            )[0]

            # Conversion des détections
            detections = sv.Detections.from_ultralytics(result)

            # Filtrage des véhicules dans la zone
           # detections = detections[
           #     polygon_zone.trigger(detections)
           # ]

            # Suivi des véhicules
            detections = byte_track.update(detections)

            detections = detections[
                detections.tracker_id != -1
            ]

            # Position du centre inférieur des véhicules
            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )

            points = view_transformer.transform_points(
                points=points
            ).astype(int)

            # Enregistrer les positions
            for tracker_id, [_, y] in zip(
                detections.tracker_id,
                points,
                strict=True
            ):

                coordinates[tracker_id].append(
                    (frame_index, y)
                )

            # Calcul des vitesses
            labels = []

            for tracker_id in detections.tracker_id:

                history = coordinates[tracker_id]

                elapsed_frames = (
                    history[-1][0] - history[0][0]
                    if len(history) > 1
                    else 0
                )

                if elapsed_frames < video_info.fps / 2:

                    labels.append(f"#{tracker_id}")

                    continue

                distance = abs(
                    history[-1][1] - history[0][1]
                )

                speed = calculate_speed(
                    distance,
                    elapsed_frames,
                    video_info.fps
                )

                labels.append(
                    f"#{tracker_id} {int(speed)} km/h"
                )

            # Dessiner les annotations
            annotated_frame = frame.copy()

            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )

            annotated_frame = box_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )

            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

            # Sauvegarder la vidéo
            sink.write_frame(annotated_frame)

            # Afficher la vidéo
            window.show(annotated_frame)

            key = window.wait_key(1)

            if not window.is_open or key == "q":
                break

            # Afficher la progression
            if frame_index % 100 == 0:

                print(
                    f"Frames traitées : {frame_index}"
                )

        window.close()

    print("Traitement terminé !")
    print("Vidéo enregistrée :", target_video_path)



# EXECUTION
if __name__ == "__main__":

    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(
        parse_optionals_as_positionals=True
    )

    auto_cli(
        main,
        as_positional=False
    )
