import json
from flask import Blueprint, Response, jsonify,request, current_app
from modules.detect_object_on_video import detect_object_on_video
from PIL import Image
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import cv2
from datetime import datetime
from io import BytesIO
from blueprints.accident.accident import accidents_collection
import cloudinary
import cloudinary.uploader

# PUBLIC BLUEPRINT...
public_bp = Blueprint('public',__name__, url_prefix='/api/v1/public')

# DETECT OBJECT ON IMAGE
def detect_object_on_image(image_file):
    # model = YOLO('./models/yolov8n.pt')
    model = YOLO('./models/i1-yolov8s.pt')
    results = model.predict(image_file)
    result = results[0]
    output = []
    for box in result.boxes:
        x1,y1,x2,y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(),2)
        output.append([
            x1,y1,x2,y2,result.names[class_id],prob
        ])
    return output

# GENERATE FRAMES
def generate_frames(path_x = ''):
    yolo_output = detect_object_on_video(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


def detect_and_save_accident(video_path):
    # Charger le modèle personnalisé qui reconnaît la classe Accident
    model_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'models', 'i1-yolov8s.pt'
        )
    )
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Impossible d'ouvrir la vidéo.")

    confidence_threshold = 0.50
    best_confidence = confidence_threshold
    best_frame = None
    frame_index = 0

    # Analyser une image sur trois
    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % 3 == 0:
            result = model.predict(
                source=frame,
                conf=confidence_threshold,
                verbose=False
            )[0]

            if result.boxes is not None and len(result.boxes) > 0:
                confidences = result.boxes.conf.tolist()
                current_confidence = max(confidences)

                if current_confidence >= best_confidence:
                    best_confidence = current_confidence
                    best_frame = frame.copy()

        frame_index += 1

    cap.release()

    # Aucun accident détecté au-dessus du seuil
    if best_frame is None:
        return None

    # Convertir l'image capturée en JPEG
    success, buffer = cv2.imencode(".jpg", best_frame)

    if not success:
        raise ValueError("Impossible de capturer l'image de l'accident.")

    # Configurer Cloudinary
    cloudinary.config(
        cloud_name=os.getenv("CLOUD_NAME"),
        api_key=os.getenv("API_KEY"),
        api_secret=os.getenv("API_SECRET")
    )

    # Enregistrer l'image capturée sur Cloudinary
    image_response = cloudinary.uploader.upload(
        BytesIO(buffer.tobytes()),
        folder="accident_frames"
    )
    image_url = image_response.get("secure_url") or image_response["url"]

    # Créer un enregistrement MongoDB
    accident = {
        "address": "Non renseignée",
        "city": "Non renseignée",
        "latitude": None,
        "longitude": None,
        "severityInPercentage": None,
        "severity": "Non évaluée",
        "detectionConfidence": round(best_confidence * 100, 2),
        "date": datetime.now(),
        "image_url": image_url
    }

    result = accidents_collection.insert_one(accident)

    return {
        "id": str(result.inserted_id),
        "confidence": round(best_confidence * 100, 2)
    }
        
# ROUTES FOR HOME...
@public_bp.route('/',methods=['GET'])
def return_home():
    return jsonify({
        "message": "Welcome to home api page."
    })

# ROUTES FOR APPLY MODEL...
@public_bp.route('/apply-model', methods=['POST'])
def detect_object():
    try:
        image_file = request.files['image']
        boxes = detect_object_on_image(Image.open(image_file.stream))
        return Response(
            json.dumps(boxes),
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({
            "status": 'error',
            'message': str(e)
        })

# ROUTES FOR UPLOAD VIDEO...
@public_bp.route('/upload-video', methods=['POST'])
@public_bp.route('/upload-video', methods=['POST'])
def api_video():
    video_file = request.files.get('image')

    if video_file is None or video_file.filename == '':
        return jsonify({
            "status": "error",
            "message": "Aucune vidéo sélectionnée."
        }), 400

    filename = secure_filename(video_file.filename)

    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)

    video_path = os.path.abspath(
        os.path.join(upload_folder, filename)
    )

    try:
        # 1. Sauvegarder la vidéo
        video_file.save(video_path)

        # 2. Analyser la vidéo et enregistrer l'accident si détecté
        accident = detect_and_save_accident(video_path)

        # 3. Renvoyer le résultat au frontend
        response_data = {
            "status": "success",
            "path": filename,
            "accident_detected": accident is not None
        }

        if accident is not None:
            response_data["accident_id"] = accident["id"]
            response_data["detection_confidence"] = accident["confidence"]
            response_data["message"] = (
                "Accident détecté et enregistré dans MongoDB."
            )
        else:
            response_data["message"] = (
                "Aucun accident détecté au-dessus du seuil."
            )

        return jsonify(response_data), 200

    except Exception as e:
        current_app.logger.exception("Erreur pendant l'analyse vidéo")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
       
# ROUTES FOR SHOW VIDEO...
@public_bp.route('/show-video/static/videos/<path>', methods=['GET'])
def show_video(path):
    print('🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥')
    final_path = 'static/videos/' + path
    return Response(generate_frames(path_x=final_path), mimetype='multipart/x-mixed-replace; boundary=frame')  
    
# ROUTES FOR WEBCAM...
@public_bp.route('/webcam', methods=['GET'])
def api_webcam():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')