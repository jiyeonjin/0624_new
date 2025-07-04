# 🚗 자율주행 자동차와 지도학습(Supervised Learning)
## 이 파일은 자율주행 자동차와 지도학습의 연관성에 대해 설명합니다.
---

## 📌 지도학습이란?

지도학습(Supervised Learning)은 **입력(Input)** 과 **정답(Output)** 쌍이 주어진 상태에서, 
모델이 이를 학습해 **보지 못한 입력에 대해 정답을 예측**하는 기계학습 방식입니다.

> 예시:  
> - 입력: 도로 이미지, 센서 값 등  
> - 출력: 조향각(왼쪽/오른쪽), 제동 여부 등

---

## 🤖 자율주행과 지도학습의 연결

자율주행 자동차는 다양한 입력(센서, 카메라, LiDAR 등)을 받아  
다음 행동(핸들 조작, 가속/감속)을 결정해야 합니다.

이때 지도학습을 활용해, 과거의 **운전자 행동 데이터**를 학습함으로써  
새로운 상황에서도 **알맞은 행동을 예측**할 수 있습니다.

### 📦 예: 도로 중심선과 차량의 거리 → 조향각 예측
| 입력 (X) | 출력 (y) |
|----------|----------|
| 차량이 중앙에서 -2만큼 벗어남 | -25도 조향 |
| 차량이 중앙에 있음 | 0도 유지 |
| 차량이 중앙에서 +2만큼 벗어남 | +25도 조향 |

---

## 🧠 예제 코드: 선형 회귀로 조향각 예측

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 🚘 입력 데이터: 도로 중심선과의 거리 (X), 조향각 (y)
X = np.array([[-2], [-1], [0], [1], [2], [3], [-3], [-1.5], [1.5], [0.5]])
y = np.array([-25, -15, 0, 15, 25, 30, -30, -18, 20, 10])

# 🧪 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📈 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 🔍 예측 및 시각화
y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color='blue', label='True')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Distance from Center')
plt.ylabel('Steering Angle')
plt.title('Supervised Learning for Autonomous Driving')
plt.legend()
plt.grid(True)
plt.show()

# 🚙 실전 예측 예시
test_input = np.array([[-1.2]])
pred_angle = model.predict(test_input)
print(f"예측 조향각 (편차 -1.2): {pred_angle[0]:.2f}도")
