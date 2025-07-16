# YOLO 가이드

## 목차
1. [YOLO란 무엇인가?](#yolo란-무엇인가)
2. [YOLO의 발전 과정](#yolo의-발전-과정)
3. [YOLO의 핵심 원리](#yolo의-핵심-원리)
4. [YOLO 버전별 특징](#yolo-버전별-특징)
5. [YOLO의 장단점](#yolo의-장단점)
6. [구현 및 사용법](#구현-및-사용법)
7. [실제 적용 사례](#실제-적용-사례)


## YOLO란 무엇인가?

YOLO (You Only Look Once)는 실시간 객체 탐지(Object Detection)를 위한 딥러닝 모델입니다. 기존의 객체 탐지 방법들과 달리, 이미지를 한 번만 보고 객체의 위치와 클래스를 동시에 예측하는 혁신적인 접근법을 제시했습니다.

### 주요 특징
- **실시간 처리**: 높은 FPS로 실시간 객체 탐지 가능
- **End-to-End 학습**: 하나의 신경망으로 전체 탐지 과정 처리
- **단일 패스**: 이미지를 한 번만 처리하여 결과 도출
- **다양한 객체 동시 탐지**: 한 이미지에서 여러 객체를 동시에 탐지

## YOLO에서 쓰이는 필수 용어

### 🎯 기본 개념

| 용어 | 설명 |
|------|------|
| **객체 탐지** | 사진에서 사람, 자동차 등을 찾아 네모 박스로 표시하는 것 |
| **바운딩 박스** | 객체를 둘러싸는 네모 박스 |
| **신뢰도** | 얼마나 확실한지 나타내는 점수 (0~1, 높을수록 확실함) |
| **클래스** | 객체의 종류 (사람, 자동차, 고양이 등) |

### 🔧 코드에서 자주 보는 용어

| 용어 | 설명 |
|------|------|
| **model.train()** | 모델 훈련시키기 |
| **model()** | 사진 분석하기 (추론) |
| **epochs** | 학습 반복 횟수 |
| **imgsz** | 이미지 크기 (보통 640) |
| **conf** | 신뢰도 임계값 (0.5 이상만 표시) |
| **results** | 분석 결과 |
| **show()** | 결과 화면에 보여주기 |

### 📈 성능 관련

| 용어 | 설명 |
|------|------|
| **FPS** | 초당 처리 가능한 이미지 수 (높을수록 빠름) |
| **mAP** | 모델 정확도 (높을수록 정확함) |
| **GPU** | 그래픽카드 (학습/추론 속도 향상) |

### 💾 파일 형식

| 용어 | 설명 |
|------|------|
| **.pt** | 훈련된 YOLO 모델 파일 |
| **.yaml** | 데이터셋 설정 파일 |
| **.jpg, .png** | 지원하는 이미지 형식 |


---
## YOLO의 발전 과정

### 타임라인
```
2015 → YOLOv1 (Joseph Redmon 등)
2016 → YOLOv2 (YOLO9000)
2018 → YOLOv3
2020 → YOLOv4 (Alexey Bochkovskiy 등)
2020 → YOLOv5 (Ultralytics)
2021 → YOLOX
2022 → YOLOv6, YOLOv7
2023 → YOLOv8
2024 → YOLOv9, YOLOv10, YOLOv11
```
---
## YOLO의 핵심 원리

### yolo 예제 코드
```
!pip install ultralytics 
from google.colab import files
from ultralytics import YOLO # COCO 사전 훈련된 YOLOv8n 모델 로드
model = YOLO("yolov8n.pt") # 모델 정보 표시 (선택사항)
model.info() # COCO8 예제 데이터셋으로 100 에포크 훈련
results = model.train(data="coco8.yaml", epochs=10, imgsz=640) # 사진 업로드하고 경로 설정
uploaded = files.upload()
image_path = list(uploaded.keys())[0] # 업로드한 이미지에 대해 YOLOv8n 모델로 추론 실행
results = model(image_path)
results[0].show()
```

> # YOLOv8 Google Colab 예제 코드 설명

### 원본 코드
```python
!pip install ultralytics  
from google.colab import files 
from ultralytics import YOLO 

### COCO 사전 훈련된 YOLOv8n 모델 로드 
model = YOLO("yolov8n.pt") 

### 모델 정보 표시 (선택사항) 
model.info() 

### COCO8 예제 데이터셋으로 100 에포크 훈련 
results = model.train(data="coco8.yaml", epochs=10, imgsz=640) 

### 사진 업로드하고 경로 설정 
uploaded = files.upload() 
image_path = list(uploaded.keys())[0] 

### 업로드한 이미지에 대해 YOLOv8n 모델로 추론 실행 
results = model(image_path) 
results[0].show()
```

## 코드 단계별 설명

### 1. 라이브러리 설치 및 import
```python
!pip install ultralytics
from google.colab import files
from ultralytics import YOLO
```
- **ultralytics**: YOLOv8 모델을 제공하는 라이브러리 설치
- **google.colab.files**: Colab에서 파일 업로드/다운로드 기능 제공
- **YOLO**: YOLOv8 모델 클래스 import

### 2. 모델 로드
```python
model = YOLO("yolov8n.pt")
```
- **yolov8n.pt**: YOLOv8 Nano 모델 (가장 가벼운 버전)
- COCO 데이터셋으로 사전 훈련된 모델 자동 다운로드
- 80개 클래스 (사람, 자동차, 동물 등) 탐지 가능

### 3. 모델 정보 표시
```python
model.info()
```
- 모델 구조, 파라미터 수, 레이어 정보 등을 출력
- 선택사항이지만 모델 이해에 도움

### 4. 모델 훈련
```python
results = model.train(data="coco8.yaml", epochs=10, imgsz=640)
```
- **data="coco8.yaml"**: COCO8 예제 데이터셋 사용 (COCO의 축소판)
- **epochs=10**: 10번 반복 훈련
- **imgsz=640**: 입력 이미지 크기 640×640 픽셀
- 실제로는 파인튜닝 과정 (이미 훈련된 모델을 추가 학습)

### 5. 이미지 업로드
```python
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
```
- Colab에서 파일 업로드 대화상자 표시
- 사용자가 선택한 이미지 파일을 업로드
- 업로드된 첫 번째 파일의 경로를 저장

### 6. 객체 탐지 실행
```python
results = model(image_path)
results[0].show()
```
- 업로드된 이미지에 대해 객체 탐지 수행
- 탐지된 객체들을 바운딩 박스와 함께 시각화


```

## 모델 종류별 사용법

### 다양한 YOLOv8 모델 크기
```python
# 모델 크기별 선택
models = {
    'nano': 'yolov8n.pt',      # 가장 빠름, 정확도 낮음
    'small': 'yolov8s.pt',     # 빠름, 정확도 보통
    'medium': 'yolov8m.pt',    # 보통 속도, 정확도 높음
    'large': 'yolov8l.pt',     # 느림, 정확도 매우 높음
    'extra_large': 'yolov8x.pt' # 가장 느림, 정확도 최고
}

# 원하는 모델 선택
model_size = 'small'  # 원하는 크기로 변경
model = YOLO(models[model_size])
```

## 결과 저장 및 내보내기

### 결과 저장
```python
# 시각화 결과 저장
results[0].save(filename='detection_result.jpg')

# 원본 이미지에 결과 그리기
annotated_img = results[0].plot()
cv2.imwrite('annotated_result.jpg', annotated_img)

# 결과 데이터를 JSON으로 저장
import json

detection_data = []
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            detection_data.append({
                'class': model.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            })

with open('detections.json', 'w') as f:
    json.dump(detection_data, f, indent=2)
```

### Google Drive에 저장
```python
from google.colab import drive

# Google Drive 마운트
drive.mount('/content/drive')

# 결과를 Drive에 저장
results[0].save(filename='/content/drive/MyDrive/yolo_result.jpg')
```


## 성능 모니터링

### 처리 시간 측정
```python
import time

start_time = time.time()
results = model(image_path)
end_time = time.time()

print(f"처리 시간: {end_time - start_time:.2f}초")
```

### GPU 사용량 확인
```python
# GPU 정보 확인
!nvidia-smi

# PyTorch에서 GPU 사용량 확인
import torch
print(f"GPU 사용 가능: {torch.cuda.is_available()}")
print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
```

# > YOLO 예제 코드 실행 결과 파일 설명
<img width="432" height="1306" alt="image" src="https://github.com/user-attachments/assets/718fa531-d836-4c84-adfb-11bac4d0de65" />


## 📁 폴더 구조

### `datasets/`
- **역할**: 훈련에 사용된 데이터셋이 저장되는 폴더
- **내용**: COCO8 예제 데이터셋의 이미지와 라벨 파일들

### `runs/`
- **역할**: 훈련 실행 결과가 저장되는 메인 폴더
- **하위 폴더**: detect, train, train2, train3, train4 등

## 🎯 `runs/detect/` 폴더
- **역할**: 추론(객체 탐지) 결과가 저장되는 폴더
- **생성 시점**: `model(image_path)` 실행 시

## 🏋️ `runs/train/` 폴더 (훈련 결과)

### 📊 성능 분석 그래프
| 파일명 | 설명 |
|--------|------|
| `BoxF1_curve.png` | F1 점수 곡선 (정밀도와 재현율의 조화평균) |
| `BoxPR_curve.png` | Precision-Recall 곡선 (정밀도-재현율 관계) |
| `BoxP_curve.png` | Precision 곡선 (정밀도 변화) |
| `BoxR_curve.png` | Recall 곡선 (재현율 변화) |

### 🔍 혼동 행렬
| 파일명 | 설명 |
|--------|------|
| `confusion_matrix.png` | 혼동 행렬 (예측 vs 실제 결과) |
| `confusion_matrix_normalized.png` | 정규화된 혼동 행렬 |

### 📈 라벨 분석
| 파일명 | 설명 |
|--------|------|
| `labels.jpg` | 데이터셋의 라벨 분포 시각화 |
| `labels_correlogram.jpg` | 라벨 간 상관관계 분석 |

### 📊 결과 데이터
| 파일명 | 설명 |
|--------|------|
| `results.csv` | 훈련 과정의 수치 결과 (에포크별 손실값, 정확도 등) |
| `results.png` | 훈련 과정 그래프 (손실값, 정확도 변화) |

### 🖼️ 훈련 배치 샘플
| 파일명 | 설명 |
|--------|------|
| `train_batch0.jpg` | 첫 번째 훈련 배치 이미지 샘플 |
| `train_batch1.jpg` | 두 번째 훈련 배치 이미지 샘플 |
| `train_batch2.jpg` | 세 번째 훈련 배치 이미지 샘플 |

### 🎯 검증 결과
| 파일명 | 설명 |
|--------|------|
| `val_batch0_labels.jpg` | 검증 배치의 실제 라벨 시각화 |
| `val_batch0_pred.jpg` | 검증 배치의 예측 결과 시각화 |

### ⚙️ 설정 파일
| 파일명 | 설명 |
|--------|------|
| `args.yaml` | 훈련에 사용된 모든 설정값 (에포크, 이미지 크기 등) |

### 🏋️ 추가 훈련 폴더
- `train2/`, `train3/`, `train4/`: 여러 번 훈련을 실행했을 때 생성
- 각각 별도의 실험 결과를 저장

## 📁 기타 폴더

### `sample_data/`
- **역할**: Colab에서 제공하는 샘플 데이터
- **내용**: 예제 파일들

### 🎯 모델 파일
| 파일명 | 설명 |
|--------|------|
| `yolo11n.pt` | YOLO11n 모델 파일 |
| `yolov8n.pt` | YOLOv8n 모델 파일 (코드에서 사용) |

## 💡 중요한 파일들

1. **results.png**: 훈련 과정을 한눈에 볼 수 있는 그래프
2. **confusion_matrix.png**: 모델 성능 분석
3. **val_batch0_pred.jpg**: 실제 예측 결과 확인
4. **weights/**: 훈련된 모델의 가중치 파일들

이 파일들을 통해 모델이 얼마나 잘 학습되었는지, 어떤 부분에서 실수하는지 등을 분석할 수 있습니다!

---


### 1. 그리드 기반 탐지
- 입력 이미지를 S×S 그리드로 분할
- 각 그리드 셀이 객체의 중심을 포함하면 해당 객체를 탐지할 책임
- 각 셀에서 B개의 바운딩 박스와 신뢰도 점수 예측

### 2. 바운딩 박스 예측
각 바운딩 박스는 다음 5개 값으로 구성:
- `x, y`: 바운딩 박스 중심 좌표
- `w, h`: 바운딩 박스 너비와 높이
- `confidence`: 객체가 있을 확률 × IoU

### 3. 클래스 예측
- 각 그리드 셀에서 C개 클래스에 대한 확률 예측
- 최종 클래스별 신뢰도 = 바운딩 박스 신뢰도 × 클래스 확률

### 4. 손실 함수
```
Loss = λ_coord × 좌표 손실 + 객체 손실 + λ_noobj × 비객체 손실 + 분류 손실
```

---
## YOLO 버전별 특징

### YOLOv5 (2020)
- **특징**:
  - PyTorch 구현
  - 사용자 친화적 인터페이스
  - AutoML 지원
  - 모델 크기별 변형 (n, s, m, l, x)
- **성능**: 140 FPS, mAP 50.7% (COCO)

### YOLOv8 (2023)
- **개선사항**:
  - Anchor-free 탐지
  - 새로운 백본 및 neck 구조
  - 향상된 데이터 증강
  - 다양한 작업 지원 (탐지, 분할, 분류)
- **성능**: mAP 53.9% (COCO)
---
## YOLO의 장단점

### 장점
- ✅ **빠른 속도**: 실시간 처리 가능
- ✅ **간단한 구조**: 단일 신경망으로 end-to-end 학습
- ✅ **전역 정보 활용**: 전체 이미지 정보를 고려한 예측
- ✅ **일반화 능력**: 다양한 도메인에서 좋은 성능
- ✅ **배경 오류 적음**: 배경을 객체로 잘못 인식하는 경우 적음

### 단점
- ❌ **작은 객체 탐지 어려움**: 그리드 기반 구조의 한계
- ❌ **겹치는 객체 처리 제한**: 한 그리드 셀에서 하나의 클래스만 예측
- ❌ **위치 정확도**: R-CNN 계열 대비 다소 낮은 위치 정확도
- ❌ **종횡비 변화**: 학습 데이터에 없는 종횡비 처리 어려움
---
## 구현 및 사용법

### 1. 환경 설정
```bash
# YOLOv5 설치
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

# YOLOv8 설치
pip install ultralytics
```

### 2. 기본 사용법

#### YOLOv5 예제
```python
import torch

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 추론
results = model('path/to/image.jpg')
results.show()  # 결과 시각화
```

#### YOLOv8 예제
```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8n.pt')

# 추론
results = model('path/to/image.jpg')
results[0].show()  # 결과 시각화
```

### 3. 학습 코드

#### YOLOv5 학습
```python
# 학습 시작
!python train.py --data dataset.yaml --cfg yolov5s.yaml --weights yolov5s.pt --batch-size 16 --epochs 100
```

#### YOLOv8 학습
```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8n.pt')

# 학습
model.train(data='dataset.yaml', epochs=100, imgsz=640)
```

### 4. 데이터셋 형식

#### YOLO 형식 라벨 파일
```
# 각 줄은 하나의 객체를 나타냄
# class_id center_x center_y width height (모든 값은 0-1로 정규화)
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

#### dataset.yaml 예제
```yaml
# 데이터셋 경로
train: path/to/train/images
val: path/to/val/images
test: path/to/test/images

# 클래스 수
nc: 80

# 클래스 이름
names: ['person', 'bicycle', 'car', 'motorcycle', ...]
```

## 성능 비교

### COCO 데이터셋 성능 (mAP@0.5:0.95)
| 모델 | mAP | FPS | 파라미터 수 |
|------|-----|-----|-------------|
| YOLOv3 | 31.0 | 20 | 62M |
| YOLOv4 | 41.2 | 62 | 64M |
| YOLOv5s | 37.4 | 140 | 7.2M |
| YOLOv5m | 45.4 | 81 | 21.2M |
| YOLOv5l | 49.0 | 61 | 46.5M |
| YOLOv5x | 50.7 | 50 | 86.7M |
| YOLOv8n | 37.3 | 150+ | 3.2M |
| YOLOv8s | 44.9 | 120+ | 11.2M |
| YOLOv8m | 50.2 | 80+ | 25.9M |
| YOLOv8l | 52.9 | 60+ | 43.7M |
| YOLOv8x | 53.9 | 50+ | 68.2M |

### 실시간 처리 성능
- **GPU**: RTX 3080 기준
- **해상도**: 640×640 입력 기준
- **배치 크기**: 1
---
## 실제 적용 사례

### 1. 자율주행
- 차량, 보행자, 신호등, 표지판 탐지
- 실시간 처리 요구사항 충족
- 다양한 날씨 및 조명 조건에서 안정적 성능

### 2. 보안 및 감시
- CCTV 영상에서 이상 행동 탐지
- 출입 통제 시스템
- 군중 밀도 모니터링

### 3. 산업 자동화
- 제조업에서 불량품 탐지
- 로봇 비전 시스템
- 품질 관리 자동화

### 4. 스포츠 분석
- 선수 및 공 추적
- 경기 통계 생성
- 실시간 경기 분석


---
## 최적화 기법

### 1. 모델 경량화
```python
# 모델 압축
model.export(format='onnx', optimize=True)  # ONNX 변환
model.export(format='torchscript')  # TorchScript 변환
model.export(format='tflite')  # TensorFlow Lite 변환
```

### 2. 추론 최적화
```python
# 혼합 정밀도 사용
model.half()  # FP16 사용

# 배치 추론
results = model(['image1.jpg', 'image2.jpg', 'image3.jpg'])
```

### 3. 하드웨어 최적화
- **GPU 최적화**: CUDA, cuDNN 활용
- **Edge 디바이스**: TensorRT, OpenVINO 사용
- **모바일**: Core ML, TensorFlow Lite 활용

---

## 트러블슈팅

### 일반적인 문제와 해결책

#### 1. 메모리 부족
```python
# 배치 크기 줄이기
batch_size = 8  # 대신 16

# 이미지 크기 줄이기
imgsz = 416  # 대신 640
```

#### 2. 학습 속도 개선
```python
# 다중 GPU 사용
device = [0, 1, 2, 3]

# 워커 수 증가
workers = 8
```

#### 3. 정확도 향상
```python
# 데이터 증강 강화
mosaic = 1.0
mixup = 0.1
copy_paste = 0.1

# 앵커 최적화
model.train(data='dataset.yaml', epochs=100, anchor_t=4.0)
```
---
## 참고 링크

### 공식 저장소
- [YOLOv5 (Ultralytics)](https://github.com/ultralytics/yolov5)
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)


### 커뮤니티
- [YOLO 공식 문서](https://docs.ultralytics.com/)
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)



