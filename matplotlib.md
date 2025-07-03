# 🧪 Python Matplotlib 기초 정리 (by W3Schools 기준)

> 📊 **Matplotlib**은 파이썬에서 가장 널리 사용되는 시각화 도구입니다.  
> 데이터를 시각적으로 표현하여 더 쉽게 이해할 수 있게 만들어 줍니다.

---

## 📌 1. Matplotlib이란?

- 데이터 시각화를 위한 **파이썬 라이브러리**
- 다양한 그래프: 선 그래프, 막대 그래프, 산점도, 히스토그램 등
- 주로 사용하는 모듈: `matplotlib.pyplot`  
  ```python
  import matplotlib.pyplot as plt

## 설치 방법

```bash
pip install matplotlib
```

## 기본 선 그래프

import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

plt.plot(x, y)
plt.show()


