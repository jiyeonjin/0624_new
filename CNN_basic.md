# 🧠 CNN (Convolutional Neural Network) 이란?

> 이미지 인식, 얼굴 검출, 자율주행까지,
> 컴퓨터가 **이미지를 이해**하는 데 쓰이는 핵심 딥러닝 구조가 바로 CNN입니다.

CNN 이미지 컨볼루션 애니메이션 링크
> 참고 링크 : https://claude.ai/public/artifacts/2cebc728-66b5-414a-9e97-991f60a2a7e1

> 참고 링크 : https://claude.ai/public/artifacts/2c09bc56-7cc3-4ea0-b3ca-7678aa107756

---

## 🔍 1. CNN이란?

CNN은 이미지 같은 **2차원 데이터**를 분석하기에 최적화된 **딥러닝 신경망 구조**입니다.

- 입력 이미지에서 **특징(윤곽, 모서리, 색감 등)**을 뽑아냅니다.
- 그 특징들을 점점 **압축하고 요약**해 최종적으로 **분류/예측**을 수행합니다.

🖼️ CNN 구조 이미지:
![CNN 구조](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)

---

## 🧱 2. CNN의 기본 구성 요소

| 구성 요소 | 설명 |
|-----------|------|
| **Convolution Layer** | 이미지를 작은 필터로 훑으며 특징을 뽑아냄 |
| **Activation Function** | ReLU를 통해 비선형성 추가 |
| **Pooling Layer** | 이미지 크기 축소, 중요한 특징만 유지 |
| **Flatten Layer** | 2D 데이터를 1D로 펼침 |
| **Fully Connected Layer** | 일반 신경망처럼 예측을 수행 |
| **Output Layer** | 결과 클래스(예: 고양이/개)를 출력 |

---

## 🌀 3. Convolution (합성곱)

- 이미지를 작은 **커널(필터)**로 슬라이딩하며 곱셈+합 연산
- 특징 맵(feature map)을 생성

🖼️ 이미지 예시:  
<img width="1564" height="676" alt="image" src="https://github.com/user-attachments/assets/dae663f1-b9a2-4d57-96dc-86401c84ee80" />



📌 수식:  
`이미지 ⨉ 커널 = 특징 맵 (Feature Map)`

---

## ⚡ 4. Activation Function (활성화 함수)

- 비선형성을 주어 복잡한 패턴을 학습 가능하게 함  
- 가장 많이 쓰는 함수: **ReLU (f(x) = max(0, x))**

🖼️ ReLU 시각화:  
![ReLU](https://upload.wikimedia.org/wikipedia/commons/6/6c/Rectifier_and_softplus_functions.svg)

---

## 📉 5. Pooling (풀링)

- 이미지 크기를 줄이고, 중요한 정보만 유지
- 일반적으로 **Max Pooling** 사용 (가장 큰 값 선택)

🖼️ Max Pooling 예시:  
![Pooling 예시](https://upload.wikimedia.org/wikipedia/commons/9/9e/Max_pooling.png)


---

## 📏 6. Flatten + Fully Connected Layer

- **Flatten**: 2D 이미지 → 1D 벡터로 변환  
- **Fully Connected**: 예측을 위한 마지막 일반 신경망 계층

🖼️ 전체 구조 흐름:  
<img width="1423" height="762" alt="image" src="https://github.com/user-attachments/assets/eb52f99a-d850-48ea-82b1-33e3918c9f70" />


---

## 🧠 7. CNN의 학습 과정

1. **Convolution → 특징 추출**
2. **Pooling → 크기 축소**
3. **Flatten → 벡터화**
4. **Dense Layer → 분류 수행**
5. **Loss Function 계산 → 오차 측정**
6. **Optimizer로 가중치 업데이트**

---

## 🛠️ 8. 예제 코드 (Keras)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```
---
# 🧩 Feature Map
# Feature Map에서 수직 필터와 수평 필터의 역할
이미지 처리 및 컨볼루션 처리 시각화 애니메이션 : 
https://claude.ai/public/artifacts/a3bda456-4c3f-4127-a921-21ad4c351c98
https://claude.ai/public/artifacts/c84d6210-cc1f-4f28-8be1-3f2150ea86e2


> CNN(합성곱 신경망)은 이미지에서 다양한 방향의 특징을 뽑아내기 위해  
> **수직 필터(Vertical Filter)**와 **수평 필터(Horizontal Filter)**를 포함한 다양한 필터를 사용합니다.

- Convolution 연산 결과로 생성되는 새로운 이미지
- 입력 이미지의 **특정한 패턴(선, 모서리, 형태 등)**을 강조한 맵
- 필터가 감지한 특징이 어디에 얼마나 강하게 나타나는지를 보여줌

---
<img width="2578" height="964" alt="image" src="https://github.com/user-attachments/assets/40c2230c-ee65-4c47-a364-0bc967b50258" />

## 수직 필터 (Vertical Filter)
<img width="2582" height="1090" alt="image" src="https://github.com/user-attachments/assets/eaa0250a-7149-4fc6-b53c-67ecc3242e2e" />

- 이미지의 **세로 방향 경계(수직선, 기둥 등)** 을 감지하는 데 사용
- 예시 필터:
  ```text
  [-1  0  1]
  [-2  0  2]
  [-1  0  1]
  ```


## 수평 필터 (Horizontal Filter)
<img width="2582" height="1094" alt="image" src="https://github.com/user-attachments/assets/0ebe6c1b-50db-470d-9b8c-959e562b5fc7" />

- 수평 필터는 이미지 내 **가로 방향 경계(수평선)** 를 감지하는 데 사용
- 예시 필터:
```text
[-1 -2 -1]
[ 0  0  0]
[ 1  2  1]
```

## 블러 필터 (Blur Filter)
<img width="2588" height="1100" alt="image" src="https://github.com/user-attachments/assets/70deec78-16dd-4ea8-a465-44e7c638c375" />

- 블러 필터는 이미지의 **디테일을 흐릿하게 만들고**, 노이즈(잡음)를 줄이는 데 사용
- 이미지의 픽셀을 **주변 픽셀과 평균** 내어 부드럽게 만듦
- 예시 필터:
```text
[1/9  1/9  1/9]
[1/9  1/9  1/9]
[1/9  1/9  1/9]
```


## 샤프닝 필터 (pening Filter) 

<img width="2600" height="1108" alt="image" src="https://github.com/user-attachments/assets/f469c759-286c-47a1-9523-2ad12bc552fc" />

- 샤프닝 필터는 이미지의 **경계와 윤곽을 뚜렷하게 강화** 시킴
- 이미지의 픽셀에서 **주변과의 차이를 강조** 하고, 흐릿한 이미지를 **선명하게 보이도록** 개선
- 예시 필터:
```text
[ 0  -1   0]
[-1   5  -1]
[ 0  -1   0]
```

## 최종 결과
<img width="2582" height="800" alt="image" src="https://github.com/user-attachments/assets/40ed407b-d8f5-4858-8dd7-8f0116739f6c" />

# CNN에서 필터가 여러 개인 이유와 구조 

> CNN(Convolutional Neural Network)은 이미지를 다양한 시각으로 분석하기 위해  
> 한 층에 **여러 개의 필터(커널)**를 동시에 사용합니다.

이 파일에서는 그 이유와 작동 방식, 결과물(Feature Map), 구조까지 설명합니다.


## 1️⃣ CNN에서 필터는 왜 여러 개 필요한가?

### 이미지엔 다양한 패턴이 존재함

이미지는 단순히 하나의 선(예: 수직선)만 있는 게 아니다.
다음과 같이 **복합적인 특징**들이 섞여 있습니다:

- 수직선 (|), 수평선 (—)
- 곡선, 점, 모서리
- 질감, 색상 변화

하지만 **필터 하나는 한 가지 방향/패턴만 감지**합니다.

---

### ✅ CNN은 어떻게 해결할까?

CNN은 **여러 개의 필터를 병렬로 사용**해,  
각 필터가 **서로 다른 특징을 감지**하도록 구성함

예를 들어:
- 필터 1: 수직선 감지
- 필터 2: 수평선 감지
- 필터 3: 곡선 감지
- ...
- 필터 64: 복합 윤곽 감지


## 2️⃣ 필터가 여러 개면 뭐가 만들어질까?

### 🎯 Feature Map 생성

각 필터는 이미지와 Convolution 연산을 수행해서  
자신만의 **Feature Map (특징 맵)**을 생성

> 즉, CNN 한 층에서 필터가 32개면, Feature Map도 32개 생성됨됨

---

### 📊 구조 흐름 예시

```text
입력 이미지: 224 x 224 x 3 (RGB)

[ 필터 1 ]  → Feature Map 1
[ 필터 2 ]  → Feature Map 2
[ 필터 3 ]  → Feature Map 3
   ...
[ 필터 64 ] → Feature Map 64

→ 최종 출력: 224 x 224 x 64
```


---
# 이외의 개념
## 📏 Stride & Padding

| 개념   | 설명 |
|--------|------|
| Stride | 필터가 한 번에 이동하는 칸 수 (기본 1칸) |
| Padding| 이미지 경계 처리 방식 (same: 크기 유지, valid: 크기 축소) |

> Stride가 클수록 출력 크기는 작아지고, Padding은 손실 없이 연산하기 위해 사용됨

---

## 🎯 Feature Map & Channel

- Feature Map: 필터 결과로 나온 특징 맵  
- Channel: 입력 이미지의 RGB 또는 CNN 내부의 깊이 차원  
- CNN에서는 여러 Feature Map이 3D 형태로 쌓임

## 🔄 Activation Function (ReLU)

ReLU(Rectified Linear Unit): 음수 제거, 양수는 그대로 유지  
비선형성을 추가해 모델의 표현력을 높임

```python
ReLU(x) = max(0, x)
```















