# YOLO (You Only Look Once) 완전 가이드

## 목차
1. [YOLO란 무엇인가?](#yolo란-무엇인가)
2. [YOLO의 발전 과정](#yolo의-발전-과정)
3. [YOLO의 핵심 원리](#yolo의-핵심-원리)
4. [YOLO 버전별 특징](#yolo-버전별-특징)
5. [YOLO의 장단점](#yolo의-장단점)
6. [구현 및 사용법](#구현-및-사용법)
7. [성능 비교](#성능-비교)
8. [실제 적용 사례](#실제-적용-사례)
9. [참고 자료](#참고-자료)

## YOLO란 무엇인가?

YOLO (You Only Look Once)는 실시간 객체 탐지(Object Detection)를 위한 딥러닝 모델입니다. 기존의 객체 탐지 방법들과 달리, 이미지를 한 번만 보고 객체의 위치와 클래스를 동시에 예측하는 혁신적인 접근법을 제시했습니다.

### 주요 특징
- **실시간 처리**: 높은 FPS로 실시간 객체 탐지 가능
- **End-to-End 학습**: 하나의 신경망으로 전체 탐지 과정 처리
- **단일 패스**: 이미지를 한 번만 처리하여 결과 도출
- **다양한 객체 동시 탐지**: 한 이미지에서 여러 객체를 동시에 탐지

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

## YOLO의 핵심 원리

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

## YOLO 버전별 특징

### YOLOv1 (2015)
- **혁신점**: 객체 탐지를 단일 회귀 문제로 재정의
- **구조**: 24개 컨볼루션 레이어 + 2개 완전연결 레이어
- **성능**: 45 FPS, mAP 63.4% (VOC 2007)
- **한계**: 작은 객체 탐지 어려움, 겹치는 객체 처리 제한

### YOLOv2 (2016)
- **개선사항**:
  - Batch Normalization 추가
  - 고해상도 분류기 사용
  - Anchor Box 도입
  - Dimension Clusters 사용
- **성능**: 67 FPS, mAP 76.8% (VOC 2007)
- **YOLO9000**: 9000개 클래스 탐지 가능

### YOLOv3 (2018)
- **주요 변화**:
  - Darknet-53 백본 사용
  - Feature Pyramid Network (FPN) 구조
  - 3개 스케일에서 예측 (13×13, 26×26, 52×52)
  - 로지스틱 분류기 사용
- **성능**: 20 FPS, mAP 57.9% (COCO)

### YOLOv4 (2020)
- **혁신 기술**:
  - CSPDarkNet53 백본
  - SPP (Spatial Pyramid Pooling)
  - PANet neck
  - Bag of Freebies (BoF) 적용
  - Bag of Specials (BoS) 적용
- **성능**: 65 FPS, mAP 43.5% (COCO)

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

### 5. 의료 영상
- 의료 영상에서 병변 탐지
- X-ray, CT 스캔 분석
- 진단 보조 도구

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

## 하이퍼파라미터 튜닝

### 주요 하이퍼파라미터
```yaml
# 학습 설정
epochs: 100
batch_size: 16
lr0: 0.01
lrf: 0.1
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# 데이터 증강
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.9
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
```

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

## 참고 자료

### 논문
- [YOLOv1: You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [YOLOv2: YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
- [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)

### 공식 저장소
- [YOLOv5 (Ultralytics)](https://github.com/ultralytics/yolov5)
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- [YOLOv4 (AlexeyAB)](https://github.com/AlexeyAB/darknet)

### 유용한 도구
- [Roboflow](https://roboflow.com/) - 데이터셋 관리 및 증강
- [Label Studio](https://labelstud.io/) - 데이터 라벨링
- [Weights & Biases](https://wandb.ai/) - 실험 추적
- [TensorBoard](https://www.tensorflow.org/tensorboard) - 시각화

### 커뮤니티
- [YOLO 공식 문서](https://docs.ultralytics.com/)
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)
- [Computer Vision 관련 논문](https://paperswithcode.com/area/computer-vision)

---

이 가이드는 YOLO에 대한 포괄적인 정보를 제공합니다. 실제 프로젝트에 적용할 때는 특정 요구사항에 맞게 조정하여 사용하세요.
