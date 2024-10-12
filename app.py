import tempfile
from flask import Flask, request, jsonify
from google.cloud import storage
import cv2
import numpy as np
import os
import torch
import subprocess  # FFmpeg 사용을 위해 필요
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm

app = Flask(__name__)

# gcs 설정
client = storage.Client()
bucket_name = 'video_uploaded'
bucket = client.get_bucket(bucket_name)

# 가중치 로드 함수
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

# FFmpeg을 사용하여 원본 동영상에서 오디오 추출
def extract_audio(video_path, output_audio_path):
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', output_audio_path, '-y']
    subprocess.run(command, check=True)

# FFmpeg을 사용하여 모자이크 처리된 영상과 오디오 합치기 및 인코딩
def merge_audio_with_video(video_path, audio_path, output_path, format_choice):
    command = ['ffmpeg', '-i', video_path, '-i', audio_path, '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', 
               '-b:a', '192k', '-shortest', output_path, '-y']
    subprocess.run(command, check=True)

# GCS에서 동영상 다운로드
def download_video(gcs_path):
    blob = bucket.blob(gcs_path)
    video_data = blob.download_as_bytes()
    return video_data

# 얼굴 탐지 및 모자이크 처리
def detect_and_mosaic_face(frame):
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

    confidence_threshold = 0.75
    nms_threshold = 0.2

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

    # 얼굴 영역에 모자이크 적용
    for b in dets:
        if b[4] < 0.5:
            continue
        b = list(map(int, b))
        x1, y1, x2, y2 = b[0:4]

        # 모자이크 적용
        face_region = frame[y1:y2, x1:x2]
        if face_region.size > 0:
            face_region = cv2.resize(face_region, (16, 16), interpolation=cv2.INTER_LINEAR)
            face_region = cv2.resize(face_region, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
            frame[y1:y2, x1:x2] = face_region

    return frame

# 모자이크 처리된 비디오 생성 및 오디오 병합
def generate_mosaic_video_with_audio(video_data, user_id, filename, format_choice):
    # 임시 파일로 원본 비디오 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_file.write(video_data)
        temp_video_file.flush()
        temp_video_path = temp_video_file.name
    
    # 오디오 추출
    with tempfile.NamedTemporaryFile(delete=False, suffix='.aac') as temp_audio_file:
        temp_audio_path = temp_audio_file.name
    extract_audio(temp_video_path, temp_audio_path)

    # 모자이크 처리된 영상 생성
    cap = cv2.VideoCapture(temp_video_path)
    frame_count = 0
    mosaic_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mosaic_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 얼굴 탐지 및 모자이크 처리
        mosaic_frame = detect_and_mosaic_face(frame)
        out.write(mosaic_frame)
        frame_count += 1

    cap.release()
    out.release()

    # 최종 비디오 파일 생성
    final_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format_choice}').name
    merge_audio_with_video(mosaic_video_path, temp_audio_path, final_video_path, format_choice)

    return final_video_path

@app.route('/apply_mosaic', methods=['POST'])
def generate_thumbnails_endpoint():
    data = request.json
    gcs_path = data.get('gcs_path')
    user_id = data.get('user_id')
    format_choice = data.get('format_choice', 'mp4')  # 사용자가 선택한 형식 (기본값은 'mp4')

    gcs_path = gcs_path.replace("gs://video_uploaded/", "")
    filename = gcs_path.split("/")[-1]
    video_data = download_video(gcs_path)

    # 모자이크 처리된 비디오 생성 및 오디오 병합
    final_video_path = generate_mosaic_video_with_audio(video_data, user_id, filename, format_choice)

    # 최종 비디오를 GCS에 업로드
    output_blob_name = f"{user_id}/processed_videos/{filename}.{format_choice}"
    output_blob = bucket.blob(output_blob_name)
    output_blob.upload_from_filename(final_video_path)

    # 임시 파일 삭제
    os.remove(final_video_path)
    os.remove(mosaic_video_path)
    os.remove(temp_video_file.name)
    os.remove(temp_audio_path)

    return jsonify({
        "message": "Processed video with mosaic and audio merged.",
        "gcs_url": f"gs://{bucket_name}/{output_blob_name}"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
