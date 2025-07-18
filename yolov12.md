# YOLOv12 완벽 정리 (2025년 기준)

YOLOv12는 객체 감지(Object Detection) 분야에서 최신 기술을 집약한 **초고속·고정확도** 알고리즘입니다. YOLO 시리즈는 "You Only Look Once"의 줄임말로, 한 번의 딥러닝 연산으로 이미지 속 객체를 빠르게 탐지합니다.

---

## 1. 📷 YOLO란?

> You Only Look Once  
이미지를 한 번만 보고도 객체를 빠르게 탐지하는 **실시간 객체 탐지 알고리즘**입니다.

- 입력 이미지 ➜ CNN 연산 ➜ 박스 + 클래스 예측
- 매우 빠르고, 모바일 기기에서도 동작 가능
- 자율주행, CCTV, 드론, 공장 자동화 등 활용

---

## 2. 🚀 YOLOv12 주요 특징

| 항목 | 설명 |
|------|------|
| ✅ **초경량화** | YOLOv11 대비 파라미터 수 최대 30% 감소 |
| ✅ **추론 속도 향상** | YOLOv8 대비 1.4배 빠른 FPS |
| ✅ **Anchor-Free 기반** | 객체 크기/비율에 상관없이 탐지 가능 |
| ✅ **Dynamic Head** | 상황별로 다른 feature 추출 (Transformer 기반) |
| ✅ **Better NMS** | `Soft-NMS`, `Weighted NMS`로 정확도 ↑ |
| ✅ **Multi-Scale Feature Fusion** | 작은 객체도 정확하게 탐지 |
| ✅ **Vision-Language 통합 지원** | "사람과 개 찾아줘" 같은 명령도 가능 (CLIP 기반)

---

## 3. 🔍 YOLOv8 vs YOLOv11 vs YOLOv12

| 항목 | YOLOv8 | YOLOv11 | YOLOv12 |
|------|--------|---------|---------|
| 출시 | 2023 | 2024 | 2025 |
| 구조 | CNN 기반 | Hybrid 구조 (CNN + Transformer) | 완전 모듈화된 구조 |
| Anchor 사용 | O (선택) | X (Anchor-Free) | X (Anchor-Free) |
| 속도 | ★★★★☆ | ★★★★☆ | ★★★★★ |
| 정확도 | ★★★★☆ | ★★★★★ | ★★★★★+ |
| 지원 모델 | n/s/m/l/x | base/custom | nano/small/medium/large/xlarge |
| 활용성 | 범용 | 고성능 연구용 | 범용 + 멀티모달 확장 가능 |
| 특징 | 가볍고 빠름 | 다중 feature map 정교화 | 확장성과 유연성 극대화 |

---

## 4. 🧠 YOLOv12 구조 한눈에 보기

```txt
[입력 이미지]
      ↓
[Backbone]
- MobileViT 또는 EfficientNetV2
- 입력 이미지에서 특징 추출

      ↓
[Neck]
- BiFPN 또는 Dynamic Feature Fusion
- 다양한 크기의 특징을 결합

      ↓
[Head]
- Anchor-Free Detection Head
- 클래스 + 바운딩 박스 예측

      ↓
[NMS (후처리)]
- Weighted NMS 또는 Soft-NMS 적용

---

## 5. 🧩 YOLOv12에서 자주 쓰이는 용어

| 용어              | 설명                                                                 |
|-------------------|----------------------------------------------------------------------|
| **Backbone**       | 이미지를 처음 받아서 특징을 추출하는 신경망 (예: MobileViT, EfficientNet) |
| **Neck**           | 다양한 크기의 특징맵을 결합하여 정보를 통합 (예: BiFPN, PANet 등)       |
| **Head**           | 객체의 위치와 클래스를 예측하는 부분                                   |
| **Anchor-Free**    | 미리 정의된 anchor 없이 객체의 중심점과 크기를 직접 예측               |
| **NMS**            | Non-Maximum Suppression. 겹치는 박스 중 가장 확실한 것만 남기는 기법    |
| **Soft-NMS**       | 박스들을 완전히 제거하지 않고 점수를 감소시켜 부드럽게 처리             |
| **Weighted NMS**   | 여러 박스를 가중 평균하여 하나의 박스로 합치는 방식                    |
| **FPS**            | Frames Per Second. 초당 처리할 수 있는 이미지 수                        |
| **mAP**            | mean Average Precision. 탐지 성능을 종합적으로 나타내는 지표             |
| **CLIP**           | 텍스트-이미지 통합 모델로, “사람 찾아줘” 같은 명령으로 탐지가 가능       |
| **vid_stride**     | 비디오에서 몇 프레임 간격으로 분석할지 설정**_**

---

##6. ⚙️ YOLOv12 기본 사용법
```
from ultralytics import YOLO

model = YOLO('yolov12n.pt')  # nano 모델 (초경량)
results = model.predict(source='your_video.mp4', save=True)
```

