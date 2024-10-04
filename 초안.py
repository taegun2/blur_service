import subprocess
import os
from flask import Flask, request, jsonify
from google.cloud import storage
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # TensorFlow 2.x 이상을 사용하는 경우
import torch
import torch.backends.cudnn as cudnn
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import io
import base64
from data import cfg_mnet, cfg_re50



app = Flask(__name__)

# Google Cloud Storage 설정
client = storage.Client()
bucket_name = 'video_uploaded'
bucket = client.get_bucket(bucket_name)

# FaceNet 모델 로드
facenet_model = load_model('facenet_keras.h5')

# 학습된 RetinaFace 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = cfg_re50  # ResNet50 기반 설정
retinaface_model = RetinaFace(cfg=cfg, phase='test')
pretrained_path = 'your_retinaface_model_path'  # 학습된 모델 경로
retinaface_model.load_state_dict(torch.load(pretrained_path, map_location=device))
retinaface_model = retinaface_model.to(device)
retinaface_model.eval()

# GPU를 사용할 경우 cudnn 최적화 설정
if torch.cuda.is_available():
    cudnn.benchmark = True

# 각 사용자의 캐시를 관리하기 위한 구조
user_cache = {}

# 사용자별 캐시 반환 함수
def get_user_cache(user_id):
    if user_id not in user_cache:
        user_cache[user_id] = {}
    return user_cache[user_id]

# GCS에서 동영상 다운로드
def download_video_from_gcs(gcs_path):
    blob = bucket.blob(gcs_path)
    video_data = blob.download_as_bytes()
    return video_data

# 오디오 추출 함수
def extract_audio(video_path, audio_output_path):
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_output_path]
    subprocess.run(command, check=True)

# 얼굴 검출 및 임베딩 추출 함수 (RetinaFace + FaceNet)
def detect_faces_and_get_embeddings(frame):
    # 이미지 전처리
    img = np.float32(frame)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    # 얼굴 검출
    with torch.no_grad():
        loc, conf, landms = retinaface_model(img)  # 추론

    scale = torch.Tensor([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
    scale = scale.to(device)
    priorbox = PriorBox(cfg, image_size=(frame.shape[0], frame.shape[1]))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    landms = landms * scale.repeat(landms.size(1) // 2)
    landms = landms.cpu().numpy()

    # 신뢰도 필터링 및 NMS 적용
    inds = np.where(scores > 0.5)[0]  # 신뢰도 임계값 적용
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)  # NMS 임계값
    dets = dets[keep, :]
    landms = landms[keep]

    face_embeddings = []

    # FaceNet을 이용한 임베딩 추출
    for i in range(dets.shape[0]):
        x1, y1, x2, y2 = dets[i][:4].astype(int)
        face_img = frame[y1:y2, x1:x2]
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        embedding = facenet_model.predict(face_img)
        face_embeddings.append({
            'box': [x1, y1, x2 - x1, y2 - y1],
            'embedding': embedding,
            'landmarks': landms[i].reshape((5, 2))  # 랜드마크 포함
        })

    return face_embeddings

# 임베딩을 비교하여 얼굴 ID 할당하는 함수
def assign_face_id(face_tracker, new_embedding, threshold=0.5):
    for face_id, tracked_face in face_tracker.items():
        distance = np.linalg.norm(tracked_face['embedding'] - new_embedding)
        if distance < threshold:
            return face_id
    
    new_face_id = len(face_tracker)
    face_tracker[new_face_id] = {'embedding': new_embedding}
    return new_face_id

# 선택된 얼굴에 모자이크 처리 및 확장자에 따라 처리
def apply_mosaic(video_data, selected_face_ids, extension, output_video_path, user_id, min_mosaic_size=10, max_mosaic_size=100):
    cap = cv2.VideoCapture(io.BytesIO(video_data))
    output_frames = []
    frame_id = 0
    temp_video_path = f"temp_output.{extension}"
    temp_audio_path = "temp_audio.aac"

    # 사용자별 캐시 가져오기
    face_tracker = get_user_cache(user_id)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_height, frame_width = frame.shape[:2]

        # 캐시된 얼굴 데이터를 사용하여 중복 검출 방지
        face_data_list = face_tracker.get(frame_id)
        if not face_data_list:
            face_data_list = detect_faces_and_get_embeddings(frame)
            face_tracker[frame_id] = face_data_list
        
        for face_data in face_data_list:
            face_id = assign_face_id(face_tracker, face_data['embedding'])
            if face_id in selected_face_ids:
                x, y, w, h = face_data['box']
                face_area = frame[y:y+h, x:x+w]
                face_area_ratio = (w * h) / (frame_width * frame_height)
                
                mosaic_size = int(min_mosaic_size + (max_mosaic_size - min_mosaic_size) * face_area_ratio)
                mosaic_size = max(min_mosaic_size, min(mosaic_size, max_mosaic_size))
                
                mosaic = cv2.resize(face_area, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
                mosaic = cv2.resize(mosaic, (w, h), interpolation=cv2.INTER_NEAREST)
                
                frame[y:y+h, x:x+w] = mosaic
        
        output_frames.append(frame)
        frame_id += 1

    cap.release()

    # 비디오로 다시 인코딩, H.264 코덱 사용
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    
    for frame in output_frames:
        out.write(frame)

    out.release()

    # 원본 비디오에서 오디오 추출
    original_video_path = f"original_video.{extension}"
    with open(original_video_path, 'wb') as f:
        f.write(video_data)
    extract_audio(original_video_path, temp_audio_path)

    # 오디오를 포함하여 최종 비디오 생성
    command = [
        'ffmpeg', '-i', temp_video_path, '-i', temp_audio_path, '-c:v', 'libx264', 
        '-c:a', 'aac', '-strict', '-2', output_video_path
    ]
    subprocess.run(command)

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    if os.path.exists(original_video_path):
        os.remove(original_video_path)

    return output_video_path

# 썸네일 생성 함수 (각 얼굴 ID당 한 번만 생성)
def generate_thumbnails(video_data, user_id):
    cap = cv2.VideoCapture(io.BytesIO(video_data))
    frame_id = 0
    thumbnails = []
    processed_face_ids = set()

    # 사용자별 캐시 가져오기
    face_tracker = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        face_data_list = face_tracker.get(frame_id)
        if not face_data_list:
            face_data_list = detect_faces_and_get_embeddings(frame)
            face_tracker[frame_id] = face_data_list
        
        for face_data in face_data_list:
            face_id = assign_face_id(face_tracker, face_data['embedding'])
            if face_id not in processed_face_ids:
                x, y, w, h = face_data['box']
                face_thumbnail = frame[y:y+h, x:x+w]
                _, buffer = cv2.imencode('.jpg', face_thumbnail)
                
                thumbnails.append({
                    'frame_id': frame_id,
                    'face_id': face_id,
                    'thumbnail': buffer.tobytes()
                })
                processed_face_ids.add(face_id)
        
        frame_id += 1
    
    cap.release()
    return thumbnails

@app.route('/generate_thumbnails', methods=['POST'])
def generate_thumbnails_endpoint():
    data = request.json
    gcs_path = data.get('gcs_path')
    user_id = data.get('user_id')  # 사용자 ID를 요청에서 받음
    video_data = download_video_from_gcs(gcs_path)
    thumbnails = generate_thumbnails(video_data, user_id)
    
    # 썸네일을 Base64로 인코딩하여 클라이언트에 전송
    thumbnails_encoded = []
    for thumbnail in thumbnails:
        thumbnail_base64 = base64.b64encode(thumbnail['thumbnail']).decode('utf-8')
        thumbnails_encoded.append({
            'frame_id': thumbnail['frame_id'],
            'face_id': thumbnail['face_id'],
            'thumbnail': thumbnail_base64
        })
    
    return jsonify({'thumbnails': thumbnails_encoded})

@app.route('/apply_mosaic', methods=['POST'])
def apply_mosaic_endpoint():
    data = request.json
    gcs_path = data.get('gcs_path')
    user_id = data.get('user_id')  # 사용자 ID를 요청에서 받음
    selected_face_ids = data.get('selected_face_ids', [])
    extension = data.get('extension', 'mp4')

    temp_output_path = f"output_with_audio.{extension}"
    
    video_data = download_video_from_gcs(gcs_path)
    output_video_path = apply_mosaic(video_data, selected_face_ids, extension, temp_output_path, user_id)

    blob = bucket.blob(f"processed/{gcs_path.split('/')[-1].split('.')[0]}.{extension}")
    with open(output_video_path, 'rb') as f:
        blob.upload_from_file(f, content_type=f'video/{extension}')
    
    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    return jsonify({'processed_video_path': blob.public_url})

if __name__ == '__main__':
    app.run(debug=True)
