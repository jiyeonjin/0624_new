# 📊 NumPy 가이드

> 이 파일은 NumPy에 대해 처음 배우는 사람을 위해 작성되었습니다.

---

## 🔷 NumPy란?

**NumPy (Numerical Python)** 는 **숫자 계산**과 **배열 연산**을 빠르고 쉽게 할 수 있도록 도와주는 파이썬 라이브러리입니다.

- 고성능 수치 계산이 가능함
- 특히 배열(Array)과 행렬(Matrix) 연산에 최적화됨
- 과학 계산, 데이터 분석, 머신러닝 등에서 기본 도구처럼 사용됨
- 내부적으로 **C언어 기반**이라 매우 빠름

---

## 💡 NumPy 설치

```bash
pip install numpy

---


## Numpy의 핵심 : 배열
import numpy as np

a = np.array([1, 2, 3])             # 1차원 배열
b = np.array([[1, 2], [3, 4]])      # 2차원 배열

np.array([1, 2, 3])           # 리스트 → 배열
np.zeros((2, 3))              # 0으로 채워진 2행 3열
np.ones((3, 2))               # 1로 채워진 배열
np.arange(0, 10, 2)           # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)          # 0~1을 5등분한 값

