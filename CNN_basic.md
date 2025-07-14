# 🧠 CNN (Convolutional Neural Network) 이란?

> 이미지 인식, 얼굴 검출, 자율주행까지,
> 컴퓨터가 **이미지를 이해**하는 데 쓰이는 핵심 딥러닝 구조가 바로 CNN입니다.

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
![Convolution 예시](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)

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
![CNN 흐름](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)

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
