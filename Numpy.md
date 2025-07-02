# 📊 NumPy 가이드

> 이 파일은 NumPy에 대해 처음 배우는 사람을 위해 작성되었습니다.

> https://www.w3schools.com/python/numpy/default.asp W3schools Numpy 튜토리얼

---

## 🔷 NumPy란?

**NumPy (Numerical Python)** 는 **숫자 계산**과 **배열 연산**을 빠르고 쉽게 할 수 있도록 도와주는 파이썬 라이브러리입니다.
"많은 숫자를 빠르고 쉽게 계산하게 도와주는 도구!"

- 고성능 수치 계산이 가능함
- 특히 배열(Array)과 행렬(Matrix) 연산에 최적화됨
- 과학 계산, 데이터 분석, 머신러닝 등에서 기본 도구처럼 사용됨
- 내부적으로 **C언어 기반**이라 매우 빠름

---

## 💡 NumPy 설치

```bash
pip install numpy
```



## 🧠 NumPy의 핵심: 배열(ndarray)
NumPy의 가장 중요한 기능은 배열(ndarray) 입니다.
기본 파이썬 리스트보다 훨씬 빠르고, 수학 계산에 최적화된 구조입니다.

import numpy as np

a = np.array([1, 2, 3])             # 1차원 배열

b = np.array([[1, 2], [3, 4]])      # 2차원 배열

## 🔑 NumPy 필수 개념 정리

### ✅ 배열 만들기

np.array([1, 2, 3])           # 리스트 → 배열

np.zeros((2, 3))              # 0으로 채워진 2행 3열

np.ones((3, 2))               # 1로 채워진 배열

np.arange(0, 10, 2)           # [0, 2, 4, 6, 8]

np.linspace(0, 1, 5)          # 0~1을 5등분한 값

### ✅ 배열 연산

a = np.array([1, 2, 3])

b = np.array([4, 5, 6])

a + b      # [5 7 9]

a - b      # [-3 -3 -3]

a * b      # [4 10 18]

a / b      # [0.25 0.4 0.5]

a ** 2     # [1 4 9] (제곱)

a * 2      # [2 4 6]


### ✅ 인덱싱 & 슬라이싱 -> 배열 안에서 찾고 자르기

a = np.array([[1, 2, 3], [4, 5, 6]])

a[0, 1]     # 0행 1열 → 2

a[:, 1]     # 모든 행의 1열 → [2 5]

a[1]        # 1행 전체 → [4 5 6]

a[0:2, 1:]  # 슬라이싱 → [[2 3], [5 6]]

### ✅ 브로드캐스팅(Broadcasting)
NumPy는 서로 다른 크기의 배열 간에도 연산을 자동으로 맞춰서 해주는 기능을 포함합니다.

a = np.array([[1, 2], [3, 4]])

b = np.array([10, 20])

print(a + b) # [[11 22]  #  [13 24]]

### ✅ 통계 함수 
a = np.array([1, 2, 3, 4, 5])

np.mean(a)   # 평균 → 3.0

np.max(a)    # 최댓값 → 5

np.min(a)    # 최솟값 → 1

np.sum(a)    # 합 → 15

np.std(a)    # 표준편차

### ✅ 랜덤 숫자 만들기

np.random.rand(2, 3)           # 0~1 사이 실수 (2x3)

np.random.randint(1, 10, (2, 2))  # 1~9 사이 정수

np.random.seed(0)              # 랜덤 고정







