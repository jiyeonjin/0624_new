# CNN 기초 용어 정리 

이 파일 CNN(합성곱 신경망)을 처음 배우는 사람들을 위해, 꼭 필요한 기본 용어를 알기 쉽게 정리한 자료입니다.

---

## 📌 1. CNN (Convolutional Neural Network)

- 이미지나 영상 분석에 많이 쓰이는 인공신경망 구조
- 사람의 시각 피질에서 영감을 받아 만들어짐
- 주요한 구성 요소: **합성곱 계층(Convolution), 풀링(Pooling), 완전연결층(Fully Connected)**

---

## 📌 2. Convolution (합성곱)

- **이미지의 특징을 뽑아내는 과정**
- 작은 필터(커널)를 이미지에 겹쳐서 스캔하며 숫자를 곱하고 더함
- 예: 가장자리(edge), 점, 선 등을 감지

## 사용 라이브러리 중 **Keras**
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```
---
## 이미지 ⨉ 커널 = 특징 맵 (Feature Map)

관련 용어  
- **Kernel (커널)**: 이미지를 훑는 작은 창
- **Stride (스트라이드)**: 커널이 움직이는 간격
- **Padding (패딩)**: 이미지 주변을 0으로 채워 사이즈 유지

---

## 📌 3. Feature Map (특징 맵)

- 합성곱 결과로 나온 이미지
- 입력 이미지보다 작고, 특정 특징(예: 윤곽선)만 남아 있음

---

## 📌 4. Activation Function (활성화 함수)

- **비선형성 부여**
- 신경망이 복잡한 패턴을 학습할 수 있게 해 줌

🔹 대표적인 함수
- **ReLU** (Rectified Linear Unit): `f(x) = max(0, x)`
  - 음수는 0으로, 양수는 그대로 출력
  - 계산 빠르고 성능 좋음

---

## 📌 5. Pooling (풀링)

- **이미지 크기를 줄이는 작업**
- 가장 중요한 특징만 남기고 나머지는 버림

🔹 종류
- **Max Pooling**: 가장 큰 값만 가져감
- **Average Pooling**: 평균값을 사용

---

## 📌 6. Flatten (평탄화)

- 2D 이미지(행렬)을 1D 벡터로 바꿈
- Fully Connected Layer에 넣기 전에 사용

---

## 📌 7. Fully Connected Layer (완전연결층)

- 뉴런들이 모두 연결된 계층
- 분류 결과를 최종 출력하는 부분

---

## 📌 8. Epoch / Batch / Iteration

| 용어        | 설명 |
|-------------|------|
| **Epoch**     | 전체 데이터셋을 1번 학습시키는 것 |
| **Batch**     | 데이터를 나눠서 학습할 때의 묶음 크기 |
| **Iteration** | 1 epoch 안에서 배치 수만큼 반복되는 학습 |

---

## 📌 9. Overfitting / Underfitting

- **Overfitting (과적합)**: 훈련 데이터에 너무 맞춰져서, 새로운 데이터에 약함
- **Underfitting (과소적합)**: 학습이 부족해서 훈련 데이터조차 잘 못 맞춤

---

## 📌 10. Dropout (드롭아웃)

- 학습 중 일부 뉴런을 랜덤하게 꺼서 과적합을 방지하는 기술

---

## 📌 11. Optimizer (최적화 기법)

- 신경망이 더 좋은 예측을 하도록 가중치를 조정해주는 알고리즘

🔹 자주 쓰이는 Optimizer
- **SGD**: 경사하강법 (기본)
- **Adam**: 더 똑똑한 경사하강법 (추천)

---

## 📌 12. Loss Function (손실 함수)

- 예측 결과와 정답의 차이를 수치로 계산
- 이 값을 줄이는 것이 학습의 목표

🔹 자주 쓰이는 손실 함수
- **MSE (Mean Squared Error)**: 회귀 문제에서 사용
- **Cross Entropy**: 분류 문제에서 사용

---

## 📌 13. Dataset (데이터셋)

- 학습에 사용하는 입력과 정답의 모음
- 예시: MNIST (숫자 손글씨), CIFAR-10 (작은 이미지)

---

## ✅ 예시: CNN 구조 흐름

```text
입력 이미지
   ↓
[Conv2D + ReLU]
   ↓
[Pooling]
   ↓
[Conv2D + ReLU]
   ↓
[Pooling]
   ↓
[Flatten]
   ↓
[Fully Connected]
   ↓
[Softmax → 예측 결과]





