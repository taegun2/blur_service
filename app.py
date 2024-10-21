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
import time
app = Flask(__name__)

# gcs 설정
client = storage.Client()
bucket_name = 'video_uploaded'
bucket = client.get_bucket(bucket_name)

# 가중치 로드 함수
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys: {}'.format(len(missing_keys)))
    print('Unused checkpoint keys: {}'.format(len(unused_pretrained_keys)))
    print('Used keys: {}'.format(len(used_pretrained_keys)))
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

# FFmpeg을 사용하여 원본 동영상에서 오디오 추출
def extract_audio(video_path, output_audio_path):
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', output_audio_path, '-y']
    subprocess.run(command, check=True)

# FFmpeg을 사용하여 모자이크 처리된 영상과 오디오 합치기 및 인코딩
def merge_audio_with_video(video_path, audio_path, output_path, format_choice, crf_value=18):
    if format_choice == 'mp4':
        output_format = 'mp4'
    elif format_choice == 'mov':
        output_format = 'mov'
    else:
        raise ValueError("지원하지 않는 형식입니다. 'mp4' 또는 'mov'만 지원합니다.")

    # FFmpeg 명령어 구성 (H.264 인코딩 및 고화질 유지)
    command = [
        'ffmpeg', '-i', video_path, '-i', audio_path,
        '-c:v', 'libx264', '-crf', str(crf_value),  # CRF를 통해 품질 설정
        '-preset', 'slow',  # 품질 우선 인코딩 속도 옵션
        '-c:a', 'aac', '-b:a', '192k',  # 오디오 코덱 및 비트레이트
        '-strict', 'experimental', '-shortest', output_path, '-y'
    ]

    subprocess.run(command, check=True)

# GCS에서 동영상 다운로드
def download_video(gcs_path):
    blob = bucket.blob(gcs_path)
    video_data = blob.download_as_bytes()
    return video_data

def detect_and_mosaic_face(frame, min_mosaic_size=10, max_mosaic_size=80):
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

        # 프레임 경계 초과 방지: 좌표를 이미지 경계 내로 조정
        x1 = max(0, min(x1, original_width))
        y1 = max(0, min(y1, original_height))
        x2 = max(0, min(x2, original_width))
        y2 = max(0, min(y2, original_height))

        # 바운딩 박스가 화면에서 차지하는 비율 계산
        face_area = (x2 - x1) * (y2 - y1)  # 얼굴 영역의 면적
        frame_area = original_width * original_height  # 전체 프레임의 면적
        face_area_ratio = face_area / frame_area  # 얼굴 영역이 차지하는 비율

        # 얼굴이 화면에서 차지하는 비율에 비례한 모자이크 크기 결정
        mosaic_size = int(min_mosaic_size + (max_mosaic_size - min_mosaic_size) * face_area_ratio * 0.5)
        mosaic_size = max(min_mosaic_size, min(mosaic_size, max_mosaic_size))

        # 모자이크 적용
        face_region = frame[y1:y2, x1:x2]
        if face_region.size > 0:
            face_region = cv2.resize(face_region, (mosaic_size, mosaic_size), interpolation=cv2.INTER_AREA)
            face_region = cv2.resize(face_region, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            frame[y1:y2, x1:x2] = face_region

    return frame
# 모자이크 처리된 비디오 생성 및 오디오 병합 (배치 처리 추가)
def generate_mosaic_video_with_audio(video_data, user_id, filename, format_choice, batch_size=24):

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
    
    # 원본 비디오의 fps 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0

    # 모자이크 처리된 비디오를 임시로 저장할 파일 생성 (항상 mp4로 저장, 변환은 나중에)
    mosaic_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mosaic_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))  # fps 적용

    batch_frames = []  # 배치에 저장할 프레임 리스트

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 배치에 프레임을 추가
        batch_frames.append(frame)

        # 배치 크기에 도달하면 처리
        if len(batch_frames) == batch_size:
            processed_batch = process_batch(batch_frames)  # 배치 단위로 처리
            for processed_frame in processed_batch:
                out.write(processed_frame)
            batch_frames = []  # 배치 초기화
            frame_count += batch_size

    # 남은 프레임이 있으면 처리
    if batch_frames:
        processed_batch = process_batch(batch_frames)
        for processed_frame in processed_batch:
            out.write(processed_frame)
        frame_count += len(batch_frames)

    cap.release()
    out.release()

    # 최종 출력 파일 경로: 확장자를 사용자가 선택한 형식으로 지정
    final_video_suffix = '.mp4' if format_choice == 'mp4' else '.mov'
    final_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=final_video_suffix).name

    # 모자이크 처리된 비디오와 오디오를 병합 및 인코딩
    merge_audio_with_video(mosaic_video_path, temp_audio_path, final_video_path, format_choice)

    return final_video_path

# 배치 단위로 프레임을 처리하는 함수
def process_batch(batch_frames):
    processed_frames = []
    for frame in batch_frames:
        # 얼굴 탐지 및 모자이크 처리 (기존 함수를 호출)
        mosaic_frame = detect_and_mosaic_face(frame)
        processed_frames.append(mosaic_frame)
    return processed_frames

@app.route('/apply_mosaic', methods=['POST'])
def generate_thumbnails_endpoint():
    data = request.json
    gcs_path = data.get('gcs_path')
    user_id = data.get('user_id')
    format_choice = data.get('format_choice', 'mp4')  # 사용자가 선택한 형식 (기본값은 'mp4')
    start = time.time()
    gcs_path = gcs_path.replace("gs://video_uploaded/", "")
    filename = gcs_path.split("/")[-1]
    video_data = download_video(gcs_path)

    # 모자이크 처리된 비디오 생성 및 오디오 병합
    final_video_path = generate_mosaic_video_with_audio(video_data, user_id, filename, format_choice)

    # 최종 비디오를 GCS에 업로드
    output_blob_name = f"{user_id}/processed_videos/{filename}"
    output_blob = bucket.blob(output_blob_name)
    output_blob.upload_from_filename(final_video_path)

    # 임시 파일 삭제 (모자이크 비디오 경로를 확인하여 삭제)
    try:
        if os.path.exists(final_video_path):
            os.remove(final_video_path)
        if 'mosaic_video_path' in locals() and os.path.exists(mosaic_video_path):
            os.remove(mosaic_video_path)
    except Exception as e:
        print(f"Error while deleting temporary files: {e}")
    end = time.time()
    total = end-start
    print(f"{total} seconds")
    return jsonify({
        "message": "Processed video with mosaic and audio merged.",
        "gcs_url": f"gs://{bucket_name}/{output_blob_name}"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
