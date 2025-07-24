# Roboflow를 활용한 이미지 라벨링, 작업 가이드

이 파일은 Roboflow를 활용하여 이미지 라벨링을 수행하고, YOLO 형식으로 데이터셋을 내보내는 과정을 단계별로 정리한 가이드입니다.

## ✅ 사전 준비

1. Roboflow 계정 생성 및 로그인:  
   https://roboflow.com 에서 계정 생성 및 로그인

2. 데이터셋 준비:  
   - 라벨링할 이미지(.jpg, .png,  등) 준비
   - 중복 이미지 제거 권장
   - 여러개의 영상을 업로드 해서 프레임 나누기

---

## 📁 프로젝트 생성

1. **[Create New Project]** 클릭
2. 프로젝트 이름 입력 (예: `0722_labeling`)
3. 프로젝트 타입 선택:
   - Project Type: `Object Detection`
   - Annotation Format: `YOLOv8` (or your preferred format)
4. [Create Project] 클릭

---

## ⬆️ 이미지 업로드

1. 상단 [Upload] 버튼 클릭
2. Drag & Drop으로 이미지 업로드
3. 중복 이미지가 있는 경우 자동으로 필터링됨
4. [Finish Uploading] 클릭 → [Annotate Now] 선택

---

## ✏️ 라벨링

1. 좌측 라벨 패널에서 원하는 클래스 생성 (ex: `traffic_light`, `lane`, `crosswalk`, `speed_sign` 등)
2. 단축키 사용 가능:
   - `B`: 박스 만들기 (bounding box)
   - `Delete`: 라벨 삭제
3. 클래스별 라벨링 가이드라인에 따라 정확하게 박스 지정
   - 예시:
     - `traffic_light`: 신호등 전체 영역
     - `lane`: 차선 (중앙선, 점선, 횡단보도 등)
     - `speed_sign`: 속도 제한 표지판
     - `lane_right_left`: 차선 방향 (좌/우회전 등)
4. 위 단계를 반복하여 여러장의 이미지 라벨랑 하기

✅ **팁:** [Settings] > [Label Assist] 기능을 통해 자동 라벨링 시도 가능

---

## 📦 라벨링 완료 후 내보내기 (Export)

1. 상단 메뉴 [Generate] 클릭
2. 원하는 버전명 설정 (예: `v1.0`)
3. Resize, Augmentation 여부 설정
4. [Generate] 클릭

---

## ⬇️ YOLO 형식으로 다운로드

1. 생성된 버전 클릭
2. [Download Dataset] 클릭
3. Format: `YOLOv8 PyTorch` 선택
4. [Download ZIP] 클릭 -> API Key 복사해두기

## 🔐 Roboflow API Key란?

**Roboflow API Key**는 사용자가 Roboflow 플랫폼에서 모델, 데이터셋, 프로젝트 등의 기능에 **프로그래밍 방식으로 접근**할 수 있도록 인증하는 **개인 고유 토큰**입니다.  
이를 통해 Python 코드 또는 외부 애플리케이션에서 Roboflow의 모델을 **불러오거나 예측 요청을 수행**할 수 있습니다.

> ✅ API Key는 비밀번호처럼 **외부에 노출되지 않도록 주의**해야 합니다.



## 🛠️ Roboflow API Key 생성 방법
1. Roboflow에 로그인  
   👉 https://roboflow.com

2. 우측 상단 프로필 아이콘 클릭  
   → `Settings` (또는 `Account`)

3. 좌측 메뉴에서 **"Roboflow API"** 선택

4. **"Create API Key"** 또는 기존 키 복사  


## 💡 API Key 사용 예시
> notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb 참고하기
```python
from roboflow import Roboflow

# [⚠️ 여기에 본인의 실제 API 키를 입력하세요]
rf = Roboflow(api_key="-----------------")

project = rf.workspace().project("0722_labeling-usrpl")
model = project.version(1).model

prediction = model.predict("test.jpg", confidence=30, overlap=50)
```


---

압축 파일에는 다음이 포함됩니다:

### 📂 YOLOv8 데이터셋 폴더 구조 설명

Roboflow에서 YOLOv8 형식으로 내보낸 데이터셋은 다음과 같은 디렉토리 구조를 가집니다:



| 경로                  | 내용 설명 |
|-----------------------|-----------|
| `data.yaml`           | 클래스 정보 및 학습/검증 이미지 경로가 포함된 설정 파일 |
| `train/images/`       | 학습에 사용할 원본 이미지들 (.jpg, .png 등) |
| `train/labels/`       | 학습 이미지에 대한 라벨 정보 (.txt 파일, YOLO 형식) |
| `valid/images/`       | 검증(Validation)에 사용할 이미지들 |
| `valid/labels/`       | 검증 이미지의 라벨 정보 |


---

## ✅ 라벨링 시 주의 사항

- [x] 클래스명 오타 없이 통일
- [x] 박스는 너무 작거나 크지 않게 조정
- [x] 흐릿한 객체는 라벨링 X -> 나중에 객체 인식의 오류 원인이 됨
- [x] 라벨링 기준에 일관성 유지
- [x] 중복 객체는 모두 개별 라벨링

---

# YOLOv8 객체 감지 모델 훈련 과정 정리

## 📋 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [Roboflow를 이용한 데이터 준비](#roboflow를-이용한-데이터-준비)
3. [YOLOv8 훈련 환경 설정](#yolov8-훈련-환경-설정)
4. [모델 훈련 실행](#모델-훈련-실행)
5. [결과 및 평가](#결과-및-평가)
6. [실제 사용 예제](#실제-사용-예제)

---

## 🎯 프로젝트 개요

### 목표
- YOLOv8을 사용한 커스텀 객체 감지 모델 훈련
- Roboflow를 활용한 효율적인 데이터 라벨링 및 관리
- 실제 영상에서의 객체 감지 성능 검증

### 사용 기술 스택
- **모델**: YOLOv8 (Ultralytics)
- **데이터 관리**: Roboflow
- **개발 환경**: Python 3.8+, CUDA (GPU 가속)
- **주요 라이브러리**: `ultralytics`, `roboflow`, `opencv-python`

---

## 📊 Roboflow를 이용한 데이터 준비

### 1. 데이터 수집 및 업로드
```bash
# 프로젝트 정보 필요 -> 샘플 코드에 이용
프로젝트명: 0722_labeling-usrpl
워크스페이스: jiyeonjin
버전: v1
```

### 2. 영상에서 이미지 추출
- **방법**: Roboflow에 영상 업로드 → 자동으로 프레임 추출
- **장점**: 
  - 자동으로 다양한 프레임 선택
  - 품질 좋은 이미지만 자동 필터링
  - 중복 프레임 제거

### 3. 라벨링 작업
#### 라벨링 도구 사용
- Roboflow의 내장 어노테이션 도구 활용
- 바운딩 박스를 이용한 객체 라벨링
- 클래스별 일관된 라벨링 기준 적용

#### 라벨링 품질 관리
```python
# 라벨링 통계 확인 예제
총 이미지 수: XXX장
총 라벨 수: XXX개
클래스별 분포:
- 클래스1: XX개
- 클래스2: XX개
```

### 4. 데이터셋 전처리
- **Train/Validation/Test 분할**: 70% / 20% / 10%
- **데이터 증강 (Augmentation)**:
  - 회전 (Rotation): ±15도
  - 밝기 조절 (Brightness): ±25%
  - 노이즈 추가 (Noise): 최대 5%
  - 아래의 이미지는 처리된 데이터셋
  - <img width="2158" height="1408" alt="image" src="https://github.com/user-attachments/assets/38803282-5e3c-4aac-b495-e3b4b3bcc1b8" />


### 5. API 키 생성
```python
API_KEY = "----------------"
```

---

## ⚙️ YOLOv8 훈련 환경 설정

### 1. 필요한 라이브러리 설치
```bash
pip install ultralytics roboflow opencv-python
```

### 2. Roboflow 데이터셋 다운로드
```python
from roboflow import Roboflow

rf = Roboflow(api_key="JwvZQEBhBR5uPrwepqQW")
project = rf.workspace("jiyeonjin").project("0722_labeling-usrpl")
dataset = project.version(1).download("yolov8")
```

### 3. 하드웨어 요구사항
- **GPU**: NVIDIA GPU (CUDA 지원)
- **메모리**: 최소 8GB RAM
- **저장공간**: 최소 10GB 여유 공간

---

## 🚀 모델 훈련 실행

### 1. 기본 훈련 명령어
```bash
yolo train data=path/to/data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### 2. 상세 훈련 설정
```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8n.pt')  # 사전 훈련된 모델 사용

# 훈련 실행
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='custom_model',
    patience=10,
    save=True,
    cache=True
)
```

### 3. 훈련 파라미터 설명
| 파라미터 | 설명 | 권장값 |
|---------|------|--------|
| `epochs` | 훈련 반복 횟수 | 100-300 |
| `imgsz` | 입력 이미지 크기 | 640 |
| `batch` | 배치 크기 | 16-32 |
| `patience` | 조기 종료 기준 | 10 |

---

## 📈 결과 및 평가

### 1. 모델 성능 지표
```python
# 모델 평가
model = YOLO('runs/train/custom_model/weights/best.pt')
metrics = model.val()

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

### 2. 훈련 결과 분석
- **Loss 그래프**: `runs/train/custom_model/results.png`
- **혼동 행렬**: `runs/train/custom_model/confusion_matrix.png`
- **PR 곡선**: `runs/train/custom_model/PR_curve.png`
- <img width="1354" height="766" alt="image" src="https://github.com/user-attachments/assets/5578823d-83ed-4f17-acb1-5452c4e63c9a" />


### 3. 모델 파일
```
runs/train/custom_model/weights/
├── best.pt      # 최고 성능 모델
├── last.pt      # 마지막 에포크 모델
└── ...
```

---

## 🎬 실제 사용 예제

### 1. YouTube 영상에서 객체 감지
```python
import cv2
import yt_dlp
from ultralytics import YOLO

def detect_objects_in_youtube_video(url, model_path):
    # YouTube 영상 다운로드
    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': 'input_video.%(ext)s',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 영상 처리
    cap = cv2.VideoCapture('input_video.mp4')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 객체 감지
        results = model(frame)
        
        # 결과 시각화
        annotated_frame = results[0].plot()
        cv2.imshow('Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 사용 예시
youtube_url = "https://www.youtube.com/watch?v=_CGb4GYHTvg"
model_path = "runs/train/custom_model/weights/best.pt"
detect_objects_in_youtube_video(youtube_url, model_path)
```

### 2. Roboflow API를 이용한 실시간 감지
```python
from roboflow import Roboflow

# Roboflow 모델 로드
rf = Roboflow(api_key="JwvZQEBhBR5uPrwepqQW")
project = rf.workspace("jiyeonjin").project("0722_labeling-usrpl")
model = project.version(1).model

# 이미지 예측
prediction = model.predict("test_image.jpg", confidence=40, overlap=30)
prediction.save("result.jpg")
```


