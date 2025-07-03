# 🧪 Python Matplotlib 가이드

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

## 1. 기본 선 그래프
```bash
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

plt.plot(x, y)
plt.show()
```

## 2. 제목 & 레이블 추가
```bash
plt.title("간단한 그래프")
plt.xlabel("X축")
plt.ylabel("Y축")
```

## 3. 마커(Marker), 색상(Color), 선 스타일(Line Style)
Matplotlib에서는 `plot()` 함수의 인자에 **마커**, **선 스타일**, **색상**을 지정하여 그래프의 모양을 꾸밀 수 있습니다.

### ✅ 기본 문법
```python
plt.plot(x, y, marker='o', linestyle='--', color='r')
```
---

### 🔵 Matplotlib Marker 종류 정리표
### 📌 마커는 어떤 역할을 하나요?

- **선 그래프에서 각 데이터 점을 명확하게 구분**해 주는 역할을 합니다.
- 특히 **여러 데이터 라인**을 하나의 그래프에 함께 그릴 때, **마커를 사용하면 시각적으로 쉽게 구별**할 수 있습니다.
- **선 없이 마커만 표시**할 수도 있습니다.

📎 예시:

plt.plot(x, y, marker='o', linestyle='None')  # 점만 표시되고 선은 없음


## 마커(Marker) 기호와 역할

| 마커 기호 | 이름            | 그래프에 표시되는 형태 / 역할             |
|-----------|------------------|--------------------------------------------|
| `'.'`     | Point             | 작은 점 (픽셀 단위)                        |
| `','`     | Pixel             | 매우 작은 점                               |
| `'o'`     | Circle            | 동그란 원형 마커                           |
| `'v'`     | Triangle Down     | 아래를 향한 삼각형                         |
| `'^'`     | Triangle Up       | 위를 향한 삼각형                           |
| `'<'`     | Triangle Left     | 왼쪽을 향한 삼각형                         |
| `'>'`     | Triangle Right    | 오른쪽을 향한 삼각형                      |
| `'1'`     | Tri Down (T형)    | T자 아래 방향 삼각형                       |
| `'2'`     | Tri Up (T형)      | T자 위 방향 삼각형                         |
| `'3'`     | Tri Left (T형)    | T자 왼쪽 방향 삼각형                       |
| `'4'`     | Tri Right (T형)   | T자 오른쪽 방향 삼각형                    |
| `'s'`     | Square            | 정사각형                                   |
| `'p'`     | Pentagon          | 오각형                                     |
| `'*'`     | Star              | 별 모양                                    |
| `'h'`     | Hexagon1          | 육각형 (스타일 1)                          |
| `'H'`     | Hexagon2          | 육각형 (스타일 2)                          |
| `'+'`     | Plus              | 십자(+) 형태                               |
| `'x'`     | X                 | X자 형태                                   |
| `'D'`     | Diamond           | 마름모 (다이아몬드)                        |
| `'d'`     | Thin Diamond      | 작은 마름모 (얇은 다이아몬드)             |
| `'|'`     | Vertical Line     | 수직 막대 (세로선)                         |
| `'_'`     | Horizontal Line   | 수평 막대 (가로선)                         |


### 🎨 Matplotlib 색상(Color) 종류 정리표

| 방식        | 예시                         | 설명                                 |
|-------------|------------------------------|--------------------------------------|
| 단축 문자    | `'r'`, `'b'`, `'k'`, `'g'` 등 | 8가지 기본 색상 (짧은 기호로 지정)     |
| 이름 지정    | `'orange'`, `'skyblue'` 등    | CSS4 색상 이름 (직관적인 이름 사용 가능) |
| HEX 코드     | `'#FF5733'`, `'#00CED1'`      | 웹 색상 코드 (HTML/CSS 스타일)         |
| RGB 튜플     | `(1.0, 0.4, 0.6)`             | 0~1 사이의 실수 3개로 구성된 RGB 값     |























