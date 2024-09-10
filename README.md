# blur_service

워크플로우:
1. 사용자가 웹 애플리케이션에서 동영상을 업로드:

사용자가 웹 애플리케이션을 통해 동영상을 업로드하면, 동영상이 **Google Cloud Storage(GCS)**에 저장됩니다.

2. GCS 경로 확보:

Cloud Run 서버가 GCS에 업로드된 동영상의 경로를 확보하고, 해당 경로를 웹 애플리케이션에 반환하거나 처리 준비를 합니다.

3. 사용자가 '분석' 버튼을 클릭:

사용자가 '분석' 버튼을 클릭하면, 해당 동영상의 GCS 경로가 Cloud Run 서버를 통해 Vertex AI로 전달됩니다.

4. Vertex AI에서 얼굴 인식 작업 수행:

Vertex AI Online Prediction API를 사용해 얼굴 인식 작업이 수행됩니다. 이 작업은 GPU를 사용해 빠르게 처리되며, 동영상에서 얼굴을 탐지합니다.
Vertex AI는 얼굴 좌표 정보와 ID 정보를 포함한 결과를 Cloud Run 서버로 반환합니다.

5. Cloud Run 서버에서 VM으로 얼굴 인식 결과 전송:

Cloud Run 서버는 Vertex AI로부터 받은 얼굴 인식 결과(좌표, ID)를 VM으로 전송합니다.

6. 썸네일 생성 (VM에서 처리):

VM 서버는 GCS에서 동영상을 다운로드하여, 얼굴 인식 결과의 좌표 정보를 기반으로 동영상 프레임에서 얼굴을 추출하고 썸네일을 생성합니다.
생성된 썸네일은 클라이언트로 전송됩니다.

7. 사용자가 블러 처리할 대상을 선택:

사용자는 웹에서 제공된 썸네일을 보고 블러 처리할 얼굴을 선택하고, 해당 얼굴 ID를 Cloud Run 서버에 전달합니다.

8. Gaussian Blur 처리 (VM에서 처리):

Cloud Run 서버는 사용자가 선택한 얼굴 ID를 다시 VM으로 전송하고, VM 서버는 해당 얼굴에 Gaussian Blur를 적용한 후, 수정된 동영상을 GCS에 다시 저장합니다.

9. 처리 완료 후 GCS 경로 반환:

VM 서버에서 비식별화가 완료된 동영상이 GCS에 저장된 후, Cloud Run 서버로 GCS 경로를 전송하고, 최종적으로 해당 경로를 클라이언트에 반환합니다.

워크플로우 요약:
Cloud Run 서버는 사용자의 동영상 업로드와 GCS 경로 확보, 그리고 Vertex AI와의 통신을 담당.
Vertex AI Online Prediction은 얼굴 인식을 빠르게 처리하고, Cloud Run 서버에 결과를 반환.
VM은 썸네일 생성과 Gaussian Blur 처리를 담당, GCS에 최종 동영상을 저장.

장점 :

GPU 성능을 최대한 활용하여 Vertex AI에서 얼굴 인식을 빠르게 처리.
썸네일 생성과 Gaussian Blur는 이미 동작 중인 VM에서 효율적으로 처리.
Cloud Run은 중간 역할을 맡아 사용자 요청을 처리하고, Vertex AI 및 VM과의 통신을 관리.
