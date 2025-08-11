# 비디오 세그멘테이션 (Video Segmentation) 

## 📖 세그멘테이션이란?

**세그멘테이션(Segmentation)**은 컴퓨터 비전에서 이미지나 비디오의 각 픽셀을 의미 있는 영역이나 객체별로 분류하는 기술입니다.

### 주요 개념

- **픽셀 단위 분류**: 이미지의 모든 픽셀을 특정 클래스(자동차, 사람, 도로 등)로 분류
- **의미론적 세그멘테이션**: 같은 클래스의 객체들을 동일하게 분류 (예: 모든 자동차를 하나의 클래스로)
- **인스턴스 세그멘테이션**: 같은 클래스 내에서도 개별 객체를 구분
- **실시간 처리**: 비디오 스트림에서 프레임별로 세그멘테이션 수행

### 활용 분야

- 🚗 **자율주행**: 도로, 차량, 보행자 인식
- 🎬 **영상 편집**: 배경 제거, 객체 추적
- 🏥 **의료 영상**: 장기, 종양 분할
- 📱 **모바일 앱**: 실시간 배경 블러, AR 필터

## 🛠️ 구현 예시: Cityscapes 모델을 이용한 비디오 세그멘테이션

다음은 Hugging Face의 SegFormer 모델을 사용하여 비디오에서 다양한 객체를 실시간으로 세그멘테이션하는 Python 코드입니다.

### 주요 특징

- **모델**: NVIDIA SegFormer-B0 (Cityscapes 데이터셋으로 학습)
- **지원 객체**: 19가지 도시 환경 객체 (도로, 차량, 사람, 건물 등)
- **실시간 처리**: GPU 가속 지원
- **시각화**: 객체별 색상 오버레이

### 코드 구조 분석

#### 1. 환경 설정 및 의존성

```python
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
```

#### 2. 모델 초기화

```python
segmenter = pipeline(
    "image-segmentation",
    model="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    device=0 if torch.cuda.is_available() else -1  # GPU 사용 가능시 GPU 활용
)
```

#### 3. 색상 매핑 시스템

각 객체 클래스별로 고유한 색상을 정의하여 시각적 구분을 용이하게 합니다:

```python
colors = {
    'road': [0, 255, 0],          # 녹색 - 도로
    'car': [0, 0, 128],           # 어두운 빨간색 - 자동차
    'person': [255, 0, 0],        # 파란색 - 사람
    'building': [128, 128, 128],   # 회색 - 건물
    # ... 더 많은 클래스들
}
```

#### 4. 비디오 처리 파이프라인

**프레임 단위 처리 과정:**

1. **프레임 읽기**: OpenCV로 비디오에서 프레임 추출
2. **색상 변환**: BGR → RGB (PIL 호환성)
3. **세그멘테이션 수행**: Hugging Face 파이프라인 실행
4. **결과 처리**: 마스크를 색상 오버레이로 변환
5. **합성 및 저장**: 원본 프레임과 오버레이 합성

```python
# 세그멘테이션 실행
results = segmenter(pil_image)

# 오버레이 생성
overlay = np.zeros_like(frame)
for result in results:
    label = result['label'].lower()
    mask = np.array(result['mask'])
    
    if label in colors:
        overlay[mask] = colors[label]

# 원본과 합성
result_frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
```

#### 5. 실시간 정보 표시

- 검출된 객체 목록 실시간 표시
- 처리 진행률 모니터링
- 프레임별 성능 추적

### 🎯 핵심 기능

#### `process_video_all_objects()` 함수
- 전체 비디오 파일을 프레임별로 세그멘테이션
- GPU 가속 처리로 성능 최적화
- 실시간 진행률 표시
- 에러 핸들링 및 복구

#### `test_single_frame()` 함수
- 디버깅 목적의 단일 프레임 테스트
- 검출 가능한 객체 종류 확인
- 모델 성능 검증

### ⚡ 성능 최적화

1. **GPU 활용**: CUDA 지원으로 처리 속도 향상
2. **배치 처리**: 가능한 경우 여러 프레임 동시 처리
3. **메모리 관리**: 대용량 비디오 처리를 위한 효율적 메모리 사용
4. **에러 복구**: 처리 실패 시 원본 프레임 유지

### 📊 지원 객체 클래스 (Cityscapes)

| 클래스 | 색상 | 설명 |
|--------|------|------|
| road | 녹색 | 도로면 |
| car | 어두운 빨간색 | 승용차 |
| person | 파란색 | 보행자 |
| building | 회색 | 건물 |
| sky | 하늘색 | 하늘 |
| vegetation | 어두운 녹색 | 식물 |
| sidewalk | 노란색 | 보도 |
| traffic light | 빨간색 | 신호등 |
| ... | ... | 총 19개 클래스 |

## 🚀 사용법

### 기본 실행

```python
# 단일 프레임 테스트
test_single_frame('/path/to/video.mp4')

# 전체 비디오 처리
process_video_all_objects('/path/to/input.mp4', '/path/to/output.mp4')
```

### 요구사항

```
torch >= 1.9.0
opencv-python >= 4.5.0
transformers >= 4.21.0
PIL (Pillow)
numpy
```

### 설치 방법

```bash
pip install torch torchvision transformers opencv-python Pillow numpy
```

## 📈 확장 가능성

- **다른 모델 적용**: DeepLab, Mask R-CNN 등
- **커스텀 클래스**: 특정 도메인에 맞는 객체 클래스 추가
- **실시간 스트리밍**: 웹캠이나 IP 카메라 입력 처리
- **후처리**: 노이즈 제거, 시간적 일관성 개선

