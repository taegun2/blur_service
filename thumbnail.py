import tempfile
from flask import Flask, request, jsonify
from google.cloud import storage
import cv2
import numpy as np
import io
import base64
import tensorflow as tf
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# gcs설정
client = storage.Client()
bucket_name = 'video_uploaded'
bucket = client.get_bucket(bucket_name)

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()

# 가중치 로드
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_to_cpu = not torch.cuda.is_available()

cfg = cfg_mnet  # ResNet50 기반 설정
retinaface_model = RetinaFace(cfg=cfg, phase='test')

# 사전 학습된 모델 로드
pretrained_path = '/home/jym542277/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth'
retinaface_model = load_model(retinaface_model, pretrained_path, load_to_cpu)

# 모델을 장치로 이동 및 평가 모드로 설정
retinaface_model = retinaface_model.to(device)
retinaface_model.eval()
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def upload_to_gcs(image, face_id, frame_count,user_id,filename):
    """썸네일 이미지를 GCS에 업로드"""
    bucket = client.bucket(bucket_name)
    blob_name = f"{user_id}/thumbnails/{filename}/face_{face_id}_frame_{frame_count}.jpg"  # GCS에 저장할 경로 및 파일명
    user_id = user_id
    blob = bucket.blob(blob_name)
    # 이미지 파일 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image_file:
        temp_image_path = temp_image_file.name
        cv2.imwrite(temp_image_path, image)  # 이미지 저장

        # GCS에 파일 업로드
        blob.upload_from_filename(temp_image_path)

    # 임시 파일 삭제
    os.remove(temp_image_path)

    # 업로드된 파일의 GCS URL 반환
    return f"gs://{bucket_name}/{blob_name}"

# RetinaFace 모델를 사용하여 얼굴 영역 반환
def detect_face(frame):
    # 원본 프레임 크기 저장
    original_height, original_width = frame.shape[:2]
    target_size = 1600
    max_size = 2150
    resize = float(target_size) / min(original_height, original_width)
    if round(resize * max(original_height, original_width)) > max_size:
        resize = float(max_size) / max(original_height, original_width)

    if resize != 1:
        frame_resized = cv2.resize(frame, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    else:
        frame_resized = frame

    # 이미지 전처리
    img = np.float32(frame_resized)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    # 얼굴 검출
    with torch.no_grad():
        loc, conf, landms = retinaface_model(img)
    
    # 박스 및 랜드마크 디코딩
    scale = torch.Tensor([frame_resized.shape[1], frame_resized.shape[0],
                          frame_resized.shape[1], frame_resized.shape[0]]).to(device)
    priorbox = PriorBox(cfg, image_size=(frame_resized.shape[0], frame_resized.shape[1]))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize  # resize 반영
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([frame_resized.shape[1], frame_resized.shape[0]] * 5).to(device)
    landms = landms * scale1 / resize  # resize 반영
    landms = landms.cpu().numpy()
    
    # 신뢰도 필터링 및 NMS 적용
    confidence_threshold = 0.5
    nms_threshold = 0.4

    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]
    return dets

# 얼굴 벡터와 ID를 장기적으로 저장할 리스트
known_faces = []  # 얼굴 특징 벡터를 저장하는 리스트
known_ids = []    # 얼굴 ID를 저장하는 리스트
def calculate_euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

# 얼굴 ID를 할당하는 함수 수정
def assign_face_id(frame, dets):
    face_images = []
    new_faces = []  # 새로 할당된 얼굴 정보를 저장할 리스트
    for b in dets:
        if b[4] < 0.5:  # 신뢰도가 낮은 경우 제외
            continue
        b = list(map(int, b))
        x1, y1, x2, y2 = b[0:4]

        # 좌표가 프레임 크기를 넘지 않도록 조정
        h, w, _ = frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 얼굴 영역 자르기
        cropped_face = frame[y1:y2, x1:x2]

        # 잘라낸 얼굴이 유효한지 확인
        if cropped_face is None or cropped_face.size == 0:
            print(f"Warning: Cropped face is empty for coordinates {x1}, {y1}, {x2}, {y2}")
            continue  # 얼굴이 유효하지 않으면 넘어감

        # 얼굴 크기를 FaceNet 입력 크기(160x160)로 조정
        try:
            resized_face = cv2.resize(cropped_face, (160, 160))
        except cv2.error as e:
            print(f"Error resizing face for coordinates {x1}, {y1}, {x2}, {y2}: {e}")
            continue  # 리사이즈 오류 발생 시 넘어감

        # FaceNet 입력 전처리
        face_image = np.float32(resized_face) / 255.0
        face_image = np.transpose(face_image, (2, 0, 1))
        face_image = torch.from_numpy(face_image).unsqueeze(0).to(device)

        # 특징 벡터 생성
        with torch.no_grad():
            embedding = facenet_model(face_image)

        # 기존 얼굴들과 비교하여 ID 할당
        assigned_id = None
        if known_faces:
            # 유클리드 거리 계산을 위해 embedding을 CPU로 옮기고 Numpy로 변환
            embedding_np = embedding.cpu().numpy()  # 변환된 Numpy 배열

            # 각 저장된 known_face와의 유클리드 거리 계산
            distances = [calculate_euclidean_distance(embedding_np, known_face.cpu().numpy()) for known_face in known_faces]
            min_distance = min(distances)

            if min_distance < 0.9:  # 일정 기준 거리 이하일 때 같은 얼굴로 판단
                assigned_id = known_ids[distances.index(min_distance)]

        # 유사한 얼굴이 없으면 새로운 ID 할당
        if assigned_id is None:
            assigned_id = len(known_ids) + 1
            known_faces.append(embedding)  # 새로운 얼굴 벡터 저장
            known_ids.append(assigned_id)  # 새로운 얼굴 ID 저장

        new_faces.append((assigned_id, (x1, y1, x2, y2)))

    return new_faces

def generate_thumbnails(video_data,user_id,filename):
    # 바이트 데이터를 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_file.write(video_data)
        temp_video_file.flush()
        temp_video_path = temp_video_file.name  # 임시 파일 경로

    cap = cv2.VideoCapture(temp_video_path)
    face_id_map = {}  # 얼굴 ID 별로 첫 번째 프레임을 저장할 딕셔너리

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        dets = detect_face(frame)
        face_images = assign_face_id(frame, dets)

        # 얼굴 ID별로 첫 번째 프레임에서 썸네일 저장
        for assigned_id, coord in face_images:
            if assigned_id not in face_id_map:
                # 새로운 ID에 대한 썸네일 저장
                x1, y1, x2, y2 = coord
                cropped_face = frame[y1:y2, x1:x2]

                # GCS에 이미지 업로드하고 URL 저장
                gcs_url = upload_to_gcs(cropped_face, assigned_id, frame_count,user_id,filename)
                print(f"gcs = {gcs_url}")
                # GCS URL을 face_id_map에 저장
                face_id_map[assigned_id] = {
                    "frame_count": frame_count,
                    "gcs_url": gcs_url
                }
        frame_count += 1

    cap.release()

    # 임시 파일 삭제
    os.remove(temp_video_path)

    # 썸네일 이미지를 Base64로 인코딩하여 반환 준비
    thumbnails = []
    for assigned_id, data in face_id_map.items():
        thumbnails.append({
            "face_id": assigned_id,
            "frame": data["frame_count"],
            "gcs_url": data["gcs_url"]
        })

    return thumbnails

# gcs에서 동영상 다운로드
def download_video(gcs_path):
    blob = bucket.blob(gcs_path)
    video_data = blob.download_as_bytes()
    return video_data



@app.route('/get_thumbnail', methods=['POST'])
def generate_thumbnails_endpoint():
    data = request.json
    gcs_path = data.get('gcs_path')
    user_id = data.get('user_id')  # 사용자 ID를 요청에서 받음
    gcs_path = gcs_path.replace("gs://video_uploaded/", "")
    filename = gcs_path.split("/")[-1]
    video_data = download_video(gcs_path)
    thumbnails = generate_thumbnails(video_data,user_id,filename)

    return jsonify({"user_id": user_id, "thumbnails": thumbnails})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
