# RunPods에서 NVIDIA PeopleNet 실행 

## 목차
1. [RunPods 개요](#runpods-개요)
2. [NGC API Key 생성](#ngc-api-key-생성)
3. [RunPods 환경 설정](#runpods-환경-설정)
4. [NVIDIA PeopleNet 코드 분석](#nvidia-peoplenet-코드-분석)
5. [실행 과정](#실행-과정)
6. [예상 결과](#예상-결과)
7. [문제 해결](#문제-해결)

---

## RunPods 개요

RunPods는 클라우드 기반 GPU 컴퓨팅 플랫폼으로, 머신러닝과 AI 모델 실행을 위한 강력한 GPU 인스턴스를 제공.

### 주요 특징
- **GPU 접근성**: 다양한 NVIDIA GPU (RTX 3080, 4090, A100 등) 제공
- **유연한 가격**: 시간당 과금 시스템으로 필요할 때만 사용
- **사전 구성된 템플릿**: PyTorch, TensorFlow 등 인기 프레임워크 지원
- **JupyterLab 통합**: 웹 기반 개발 환경 제공
- **Docker 지원**: 커스텀 환경 구성 가능

### RunPods 시작하기

1. **계정 생성**
   - [RunPods 웹사이트](https://www.runpod.io)에서 계정 생성
   - 결제 방법 등록 (크레딧 카드 또는 PayPal)

2. **Pod 생성**
   - "Deploy" 버튼 클릭
   - GPU 타입 선택 (예: RTX 3090)
   - 템플릿 선택 (PyTorch 2.4 CUDA 12.4 권장)
   - 포트 설정: 8888 (JupyterLab), 22 (SSH)

3. **JupyterLab 접속**
   - Pod 생성 후 "Connect" 버튼 클릭
   - "HTTP Services" → "Jupyter Lab" 선택
   - 웹 브라우저에서 JupyterLab 환경 접속

---

## NGC API Key 생성

NVIDIA NGC (NVIDIA GPU Cloud)는 AI 모델과 컨테이너를 제공하는 플랫폼입니다. PeopleNet 모델을 다운로드하려면 API Key가 필요합니다.

### 단계별 가이드

1. **NGC 계정 생성**
   - [NVIDIA NGC 웹사이트](https://ngc.nvidia.com) 방문
   - "Sign Up" 클릭하여 계정 생성
   - 이메일 인증 완료

2. **API Key 생성**
   - NGC에 로그인 후 우측 상단 프로필 아이콘 클릭
   - "Setup" 또는 "API Key" 메뉴 선택
   - "Generate API Key" 클릭
   - 키 이름 입력 (예: "RunPods-PeopleNet")
   - 만료 기간 설정 (권장: Never expires)
   - "Generate API Key" 클릭

3. **API Key 보안**
   - 생성된 API Key를 안전한 곳에 저장
   - 타인과 공유하지 말 것
   - 필요시 새로운 키 생성 가능

### API Key 사용법

NGC CLI 설정 시 다음 정보 입력:
- **API Key**: 생성한 API Key
- **CLI output format**: ascii (기본값)
- **Org**: nvidian/nim (기본값)
- **Team**: no-team (기본값)

---

## RunPods 환경 설정

### 1. 기본 패키지 설치

```bash
# JupyterLab 셀에서 실행 (! 붙이기)
!apt update && apt install -y unzip wget ffmpeg
```

### 2. NGC CLI 설치

```bash
# NGC CLI 다운로드
!wget -q https://ngc.nvidia.com/downloads/ngccli_reg_linux.zip

# 압축 해제
!unzip -o ngccli_reg_linux.zip

# 권한 부여
!chmod +x ngc-cli/ngc
```

### 3. NGC CLI 설정

```bash
# 설정 실행
!./ngc-cli/ngc config set
```

**설정 시 입력사항:**
- Enter API key: [생성한 NGC API Key]
- Enter CLI output format: ascii
- Enter org: nvidian/nim
- Enter team: no-team

### 4. PeopleNet 모델 다운로드

```bash
# PeopleNet 모델 다운로드
!./ngc-cli/ngc registry model download-version nvidia/tao/peoplenet:pruned_quantized_decrypted_v2.3.4
```

### 5. Python 패키지 설치

```bash
# 필요한 패키지 설치
!pip install onnxruntime yt-dlp opencv-python numpy
```

---

## NVIDIA PeopleNet 코드 분석

제공된 코드는 NVIDIA PeopleNet을 사용한 사람 검출 시스템의 디버깅 버전입니다.

### 주요 클래스: `DebugNVIDIAPeopleNet`

#### 1. 초기화 및 모델 로드

```python
class DebugNVIDIAPeopleNet:
    def __init__(self):
        # 모델 경로 설정
        self.model_path = "/workspace/peoplenet_vpruned_quantized_decrypted_v2.3.4/resnet34_peoplenet_int8.onnx"
        self.classes = ['person', 'bag', 'face']  # 검출 가능한 클래스
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # 시각화 색상
```

**기능:**
- PeopleNet ONNX 모델 경로 설정
- 검출 가능한 3개 클래스 정의 (사람, 가방, 얼굴)
- 각 클래스별 시각화 색상 정의

#### 2. 모델 설정 및 검증

```python
def setup_model(self):
    # ONNX Runtime으로 모델 로드
    providers = ['CPUExecutionProvider']
    self.session = ort.InferenceSession(self.model_path, providers=providers)
    
    # 입출력 정보 확인
    input_info = self.session.get_inputs()[0]
    output_info = self.session.get_outputs()
```

**기능:**
- ONNX Runtime을 사용한 모델 로드
- CPU 실행 환경 설정 (GPU 사용 시 'CUDAExecutionProvider' 추가 가능)
- 모델의 입출력 구조 분석

#### 3. 전처리 함수

```python
def preprocess_frame(self, frame):
    # 960x544로 리사이즈 (PeopleNet 입력 크기)
    resized = cv2.resize(frame, (960, 544))
    
    # BGR → RGB 변환
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # 정규화 (0-255 → 0-1)
    normalized = rgb_frame.astype(np.float32) / 255.0
    
    # HWC → CHW 형태 변환 (Height-Width-Channel → Channel-Height-Width)
    chw_frame = np.transpose(normalized, (2, 0, 1))
    
    # 배치 차원 추가 (1, 3, 544, 960)
    batch_frame = np.expand_dims(chw_frame, axis=0)
```

**기능:**
- OpenCV 프레임을 PeopleNet 입력 형식으로 변환
- 크기 조정, 색상 공간 변환, 정규화, 차원 재배열

#### 4. 검출 함수

```python
def detect_people(self, frame, debug=True):
    # 전처리
    input_data = self.preprocess_frame(frame)
    
    # 모델 추론
    outputs = self.session.run(self.output_names, {self.input_name: input_data})
    
    # 후처리
    detections = self.postprocess_debug(outputs, frame.shape, debug=debug)
```

**기능:**
- 프레임 전처리 → 모델 추론 → 결과 후처리 파이프라인
- 디버그 모드에서 상세한 로그 출력

#### 5. 후처리 함수

```python
def postprocess_debug(self, outputs, original_shape, debug=True):
    predictions = outputs[0]  # 모델 출력 (3, 34, 60)
    
    # 각 클래스별 처리
    for class_idx in range(min(num_classes, len(self.classes))):
        class_pred = predictions[class_idx]  # (34, 60) 그리드
        
        # 다양한 임계값으로 검출 시도
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for threshold in thresholds:
            high_positions = np.where(class_pred > threshold)
```

**기능:**
- 모델 출력을 실제 바운딩 박스로 변환
- 다양한 신뢰도 임계값으로 검출 결과 확인
- NMS (Non-Maximum Suppression)를 통한 중복 제거

#### 6. 유틸리티 함수들

- **`download_youtube_video()`**: yt-dlp를 사용한 YouTube 동영상 다운로드
- **`test_video_frames()`**: 비디오에서 특정 프레임들을 추출하여 테스트
- **`draw_detections()`**: 검출 결과를 이미지에 시각화
- **`simple_nms()`**: 중복된 검출 결과 제거
- **`calculate_iou()`**: IoU (Intersection over Union) 계산

### 코드의 특징

1. **디버깅 중심**: 모든 단계에서 상세한 로그 출력
2. **에러 처리**: 모델 파일 누락, 다운로드 실패 등 다양한 상황 대응
3. **유연성**: 다양한 임계값과 설정으로 최적화 가능
4. **시각화**: 검출 결과를 이미지에 바로 표시

---

## 실행 과정

### 1. JupyterLab에서 코드 실행

1. **새 노트북 생성**
   - JupyterLab에서 "Python 3 (ipykernel)" 노트북 생성

2. **환경 설정 코드 실행**
   ```python
   # 첫 번째 셀
   !apt update && apt install -y unzip wget ffmpeg
   !pip install onnxruntime yt-dlp opencv-python numpy
   ```

3. **NGC CLI 설정**
   ```python
   # 두 번째 셀
   !wget -q https://ngc.nvidia.com/downloads/ngccli_reg_linux.zip
   !unzip -o ngccli_reg_linux.zip
   !chmod +x ngc-cli/ngc
   !./ngc-cli/ngc config set
   ```

4. **모델 다운로드**
   ```python
   # 세 번째 셀
   !./ngc-cli/ngc registry model download-version nvidia/tao/peoplenet:pruned_quantized_decrypted_v2.3.4
   ```

5. **메인 코드 실행**
   - 제공된 전체 Python 코드를 새 셀에 복사하여 실행

### 2. 실행 단계별 진행

1. **모델 초기화**
   - PeopleNet 모델 파일 검색 및 로드
   - ONNX Runtime 세션 생성
   - 모델 구조 분석 및 테스트

2. **YouTube 동영상 다운로드**
   - 지정된 YouTube URL에서 동영상 다운로드
   - 720p 이하 품질로 제한하여 처리 속도 향상

3. **프레임 단위 테스트**
   - 100프레임 간격으로 5개 프레임 추출
   - 각 프레임에서 사람/가방/얼굴 검출
   - 신뢰도와 바운딩 박스 정보 출력

---

## 예상 결과

### 1. 성공적인 실행 로그

```
🚀 디버깅 NVIDIA PeopleNet 시작...
📁 모델 경로 확인: /workspace/peoplenet_vpruned_quantized_decrypted_v2.3.4/resnet34_peoplenet_int8.onnx
✅ 모델 로드 성공!
📊 입력: input_1, 형태: [1, 3, 544, 960]
📊 출력 개수: 1
🧪 모델 테스트 중...
✅ 테스트 성공!
   출력 0: (1, 3, 34, 60), 범위 [0.000, 0.987]

📺 YouTube 다운로드: https://www.youtube.com/watch?v=SzRzYvQq0aQ
✅ 다운로드 완료: /workspace/debug_input_video.mp4

🎬 5개 프레임 테스트 시작...

🎯 프레임 0 테스트:
🔍 입력 프레임: (720, 1280, 3)
📊 전처리 완료: (1, 3, 544, 960)
🤖 추론 완료
   출력 0: (1, 3, 34, 60)
   범위: [0.0000, 0.8234]
   평균: 0.0123
   값 분포: >0.1(45), >0.3(12), >0.5(3)
🔍 후처리 시작: (1, 3, 34, 60)
📊 그리드: 3 클래스, 34x60
   person 최대값: 0.8234
   person 임계값 0.1: 23개 후보
🎯 최종 검출: 2개
   - person: 0.823
   - person: 0.567
✅ 2개 검출 성공!
```

### 2. 검출 결과

각 프레임에서 다음과 같은 정보가 출력됩니다:

- **검출된 객체 수**: 프레임당 0-5개 정도
- **검출 클래스**: person, bag, face
- **신뢰도**: 0.0-1.0 범위의 점수
- **바운딩 박스**: 객체의 위치 좌표

### 3. 성능 지표

- **처리 속도**: GPU 사용 시 프레임당 50-100ms
- **검출 정확도**: 사람 검출 약 85-95%
- **메모리 사용량**: 약 2-3GB RAM

---

## 문제 해결

### 1. 일반적인 오류

#### NGC API Key 관련
```
❌ NGC CLI 설정 실패
```
**해결책:**
- NGC 웹사이트에서 API Key 재생성
- 올바른 형식으로 입력 확인
- 네트워크 연결 상태 확인

#### 모델 다운로드 실패
```
❌ 모델 파일이 없습니다
```
**해결책:**
```bash
# 모델 파일 수동 확인
!find /workspace -name "*peoplenet*" -type f
!ls -la /workspace/

# 다시 다운로드 시도
!./ngc-cli/ngc registry model download-version nvidia/tao/peoplenet:pruned_quantized_decrypted_v2.3.4
```

#### 패키지 설치 오류
```
❌ pip install 실패
```
**해결책:**
```bash
# pip 업그레이드
!pip install --upgrade pip

# 개별 패키지 설치
!pip install onnxruntime
!pip install yt-dlp
!pip install opencv-python
!pip install numpy
```

### 2. 성능 최적화

#### GPU 사용 설정
```python
# CPU 대신 GPU 사용
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
self.session = ort.InferenceSession(self.model_path, providers=providers)
```

#### 배치 처리
```python
# 여러 프레임 동시 처리
def batch_detect(self, frames):
    batch_input = np.stack([self.preprocess_frame(frame) for frame in frames])
    outputs = self.session.run(self.output_names, {self.input_name: batch_input})
    return outputs
```

### 3. 메모리 최적화

```python
# 큰 비디오 파일 처리 시
import gc

def process_large_video(self, video_path):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 프레임 처리
        detections = self.detect_people(frame, debug=False)
        
        # 메모리 정리
        if frame_count % 100 == 0:
            gc.collect()
    
    cap.release()
```

### 4. 디버깅 팁

1. **로그 레벨 조정**
   ```python
   # 상세 로그 비활성화
   detections = self.detect_people(frame, debug=False)
   ```

2. **중간 결과 저장**
   ```python
   # 전처리 결과 저장
   np.save('preprocessed_frame.npy', input_data)
   
   # 모델 출력 저장
   np.save('model_output.npy', outputs)
   ```

3. **시각화 확인**
   ```python
   # 검출 결과 이미지 저장
   result_frame = self.draw_detections(frame, detections)
   cv2.imwrite('detection_result.jpg', result_frame)
   ```

---

## 추가 정보

### RunPods 비용 최적화
- **자동 종료**: 사용하지 않을 때 Pod 자동 종료 설정
- **적절한 GPU 선택**: 작업에 맞는 최소 사양 GPU 선택
- **스토리지 관리**: 불필요한 파일 정기적 삭제

### 확장 가능성
- **실시간 스트리밍**: 웹캠이나 RTSP 스트림 처리
- **배치 처리**: 대량의 비디오 파일 자동 처리
- **API 서버**: Flask/FastAPI를 사용한 REST API 구현
- **모델 최적화**: TensorRT나 OpenVINO로 추가 최적화
