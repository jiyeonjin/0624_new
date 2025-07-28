# YOLO 객체 탐지와 Tensor 개념 정리

## 🎯 YOLO 코드 분석

### 1. 환경 설정 및 데이터 준비

```python
from google.colab import drive
drive.mount('/content/drive')

# ZIP 파일을 코랩으로 복사
!cp "/content/drive/MyDrive/6_23_Lesson/dataset.zip" "/content/"
# 압축 해제
!unzip -o /content/dataset.zip -d /content/
```
**설명**: Google Drive에서 학습 데이터셋을 Colab 환경으로 복사하고 압축을 해제합니다.

### 2. YOLO 설치 및 모델 로드

```python
!pip install ultralytics
from ultralytics import YOLO

# 이미 학습된 모델 사용
model = YOLO('/content/dataset/best.pt')
```
**설명**: 
- `ultralytics`: YOLO v8/v9의 공식 라이브러리
- `best.pt`: 사전 학습된 가중치 파일 (PyTorch 모델)

### 3. 테스트 영상 다운로드

```python
!pip install yt-dlp
!yt-dlp -f 'best[height<=720]' -o '/content/test_video.%(ext)s' 'https://www.youtube.com/watch?v=AxLmroTo3rQ'

# 다운로드된 파일 찾기
video_files = glob.glob('/content/test_video.*')
video_path = video_files[0]
```
**설명**: YouTube에서 720p 해상도의 영상을 다운로드하여 객체 탐지 테스트용으로 사용합니다.

### 4. 객체 탐지 실행

```python
# 추론 실행
results = model(video_path)
# 결과 표시 (영상의 경우 첫 번째 프레임만)
results[0].show()
```
**설명**: 
- `model(video_path)`: 영상에 대해 객체 탐지 수행
- `results[0].show()`: 탐지 결과를 시각화하여 출력

### 5. 모델 성능 평가

```python
# dataset.yaml 파일 수정 (경로 문제 해결)
yaml_fix = '''path: /content/dataset
train: train/images
val: valid/images
names:
  0: lane
  1: traffic_sign
nc: 2'''

with open('/content/dataset/dataset_fixed.yaml', 'w') as f:
    f.write(yaml_fix)

# 성능 평가 실행
metrics = model.val(data='/content/dataset/dataset_fixed.yaml')
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

**성능 지표 설명**:
- **mAP50**: IoU 임계값 0.5에서의 평균 정밀도
- **mAP50-95**: IoU 임계값 0.5~0.95 범위의 평균 정밀도
- **Precision**: 정밀도 (예측한 것 중 맞은 비율)
- **Recall**: 재현율 (실제 객체 중 찾아낸 비율)

---

## 🧮 Tensor란 무엇인가?

### 정의
**Tensor(텐서)**는 다차원 배열을 일반화한 수학적 객체로, 딥러닝에서 데이터를 표현하는 기본 단위입니다.

### 차원별 분류

| 차원 | 이름 | 형태 | 예시 |
|------|------|------|------|
| 0차원 | 스칼라(Scalar) | 단일 값 | `5` |
| 1차원 | 벡터(Vector) | 1D 배열 | `[1, 2, 3]` |
| 2차원 | 행렬(Matrix) | 2D 배열 | `[[1,2], [3,4]]` |
| 3차원+ | 텐서(Tensor) | 다차원 배열 | `[[[1,2],[3,4]], [[5,6],[7,8]]]` |

### 실제 활용 예시

#### 1. 이미지 데이터
```python
# RGB 이미지: (높이, 너비, 채널)
image_tensor = torch.randn(224, 224, 3)  # 224x224 픽셀, RGB 3채널
```

#### 2. 배치 데이터
```python
# 배치 이미지: (배치크기, 채널, 높이, 너비)
batch_tensor = torch.randn(32, 3, 224, 224)  # 32장의 이미지
```

#### 3. YOLO에서의 텐서 활용
```python
# YOLO 입력: (배치, 채널, 높이, 너비)
input_tensor = torch.randn(1, 3, 640, 640)

# YOLO 출력: (배치, 예측박스수, 클래스수+5)
# 5 = x, y, w, h, confidence
output_tensor = torch.randn(1, 25200, 7)  # 2클래스 + 5 = 7
```

### PyTorch에서 텐서 조작

#### 기본 생성
```python
import torch

# 다양한 텐서 생성 방법
zeros = torch.zeros(2, 3)          # 0으로 채운 2x3 텐서
ones = torch.ones(2, 3)            # 1로 채운 2x3 텐서
random = torch.randn(2, 3)         # 정규분포 랜덤 2x3 텐서
from_list = torch.tensor([[1, 2], [3, 4]])  # 리스트에서 생성
```

#### 형태 변환
```python
x = torch.randn(4, 6)
print(x.shape)  # torch.Size([4, 6])

# reshape: 형태 변경
y = x.reshape(2, 12)
print(y.shape)  # torch.Size([2, 12])

# view: 메모리 공유하며 형태 변경
z = x.view(24)
print(z.shape)  # torch.Size([24])
```

#### 연산
```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 기본 연산
add_result = a + b        # [5, 7, 9]
mul_result = a * b        # [4, 10, 18]
dot_result = torch.dot(a, b)  # 32 (내적)

# 행렬 곱셈
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.matmul(A, B)    # (3, 5) 크기의 결과
```

### GPU 활용
```python
# GPU로 이동 (CUDA 사용 가능한 경우)
if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_tensor = torch.randn(1000, 1000).to(device)
else:
    device = torch.device('cpu')
    cpu_tensor = torch.randn(1000, 1000)
```

### 딥러닝에서의 역할

1. **데이터 표현**: 이미지, 텍스트, 음성 등 모든 데이터를 텐서로 변환
2. **모델 파라미터**: 가중치와 편향을 텐서로 저장
3. **연산 최적화**: GPU 병렬 처리를 통한 빠른 계산
4. **자동 미분**: 역전파 알고리즘을 위한 그래디언트 자동 계산

### 메모리 효율성
```python
# 메모리 공유 확인
x = torch.randn(2, 3)
y = x.view(6)
print(x.data_ptr() == y.data_ptr())  # True (같은 메모리 공유)

# 새로운 메모리 할당
z = x.clone()
print(x.data_ptr() == z.data_ptr())  # False (다른 메모리)
```

---

## 🔗 YOLO와 텐서의 연관성

YOLO 모델에서 텐서는 다음과 같이 활용됩니다:

1. **입력 이미지**: `(1, 3, 640, 640)` 형태의 4차원 텐서
2. **특징 맵**: 컨볼루션 레이어를 거치며 생성되는 다차원 텐서들
3. **예측 결과**: 바운딩 박스 좌표와 클래스 확률을 담은 텐서
4. **손실 계산**: 예측값과 실제값 간의 차이를 텐서 연산으로 계산

이처럼 텐서는 딥러닝의 핵심 데이터 구조로, YOLO와 같은 객체 탐지 모델의 모든 과정에서 중요한 역할을 합니다.
