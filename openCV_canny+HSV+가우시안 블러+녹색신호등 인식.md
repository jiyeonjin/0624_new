# openCV_canny+HSV+가우시안+녹색신호등 코드

## 기존 신호등 인식 코드
> https://colab.research.google.com/drive/1QftS54mhghhs3xLfIMBABcxz6Oq0q5lK#scrollTo=-NfuZzAOnlsv

## 기존 코드 분석 결과

### 🚦 신호등 인식 프로그램 (Python + OpenCV)

> 초보자를 위한 신호등 인식 프로젝트 정리  
> - OpenCV, NumPy, Matplotlib, Google Colab 환경 사용  
> - 색상 필터링 + Canny Edge + 윤곽선 분석 기반

---

### 📌 주요 기능 요약

- 빨강/노랑/초록/파랑 신호등 색상만 필터링
- Canny 엣지 검출로 윤곽선 추출
- 원형성, 위치, 면적 기준으로 신호등 판단
- 사각형으로 검출 결과 시각화
- 파라미터 조정 및 비교 기능 포함

---

### ✅ 코드 구조 및 설명

#### 1️⃣ 라이브러리 불러오기

```python
import cv2 # 이미지 처리용
import numpy as np # 이미지 배열 처리
import matplotlib.pyplot as plt # 이미지 시각화
from google.colab import files
from PIL import Image
import io # 이미지 업로드 및 읽기
```

#### 2️⃣ 신호등 검출 함수: `detect_traffic_light_canny()`

사진에서 신호등(빨강/노랑/초록/파랑)을 찾아내는 핵심 함수입니다.  
HSV 색상 필터링 + Canny 엣지 검출 + 윤곽선 조건 판단을 통해 신호등만 골라냅니다.

---

#### 처리 순서 요약

| 단계 | 설명 | 함수/기능 |
|------|------|-----------|
| Step 0 | HSV 색상 필터링 (빨강/노랑/초록/파랑) | `cv2.cvtColor`, `cv2.inRange`, `cv2.bitwise_or` |
| Step 1 | 필터링 결과를 흑백으로 변환 | `cv2.cvtColor(..., COLOR_BGR2GRAY)` |
| Step 2 | GaussianBlur로 노이즈 제거 | `cv2.GaussianBlur()` |
| Step 3 | Canny 엣지 검출 (경계선 강조) | `cv2.Canny()` |
| Step 4 | 윤곽선(contour) 찾기 | `cv2.findContours()` |
| Step 5 | 윤곽선 필터링 (면적, 위치, 비율, 원형성) | `cv2.contourArea()`, `cv2.boundingRect()` 등 |

---

##### 🎯 Step 5 – 조건 필터링 기준

| 조건명 | 기준 | 설명 |
|--------|------|------|
| 면적 (`area`) | `min_area` ~ `max_area` | 너무 작거나 너무 큰 객체는 제외 |
| 위치 (`y`) | 상단 70% 이내만 허용 | 하단 30%에 있는 객체는 제외 (신호등은 보통 위에 있음) |
| 종횡비 (`aspect_ratio = w/h`) | ≤ 0.8 | 신호등은 보통 세로로 김 |
| 원형성 (`circularity`) | ≥ 0.25 (기본값) | `4π × 면적 / 둘레²` 값. 원형에 가까울수록 1에 가까움 |

---

#### ✅ 반환값

- `traffic_lights`: 신호등으로 판단된 `(x, y, w, h)` 좌표 리스트
- `edges`: 중간에 계산된 Canny 엣지 이미지

```python
return traffic_lights, edges
```

#### 3️⃣ 결과 시각화 함수: `draw_detections()`

신호등으로 검출된 객체들에 대해 이미지 위에 **사각형과 라벨 텍스트를 그리는 함수**입니다.  
최종 결과 이미지를 사용자에게 시각적으로 보여줄 수 있게 만듭니다.

---

#### 🧠 함수 정의

```python
def draw_detections(image, detections):
    result = image.copy()
    for i, (x, y, w, h) in enumerate(detections):
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2) # 사각형 색상 - 초록색
        cv2.putText(result, f'Traffic Light {i+1}', (x, y-10), # 텍스트 위치 - 박스 위쪽 여백
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) # 폰트, 텍스트 크기
    return result
```
#### 4️⃣ 색상 필터링 비교 함수: `compare_with_without_color()`

> 색상 필터링을 **사용한 경우와 사용하지 않은 경우**의 신호등 인식 성능을 비교하는 함수입니다.

---

##### 📋 주요 목적

| 항목 | 설명 |
|------|------|
| 테스트 | 같은 이미지로 두 방식(Canny만 vs 색상 필터링+Canny)을 비교 |
| 시각화 | 두 결과를 한 화면에 나란히 시각화 |
| 평가 | 어떤 방법이 더 정확하게 검출하는지 판단 |

---

##### 처리 순서 요약

1. Google Colab에서 이미지 업로드
2. 색상 필터링 **사용하지 않음** → `detect_traffic_light_canny(..., use_color_filter=False)`
3. 색상 필터링 **사용함** → `detect_traffic_light_canny(..., use_color_filter=True)`
4. `draw_detections()`로 결과 이미지 생성
5. `matplotlib`으로 두 결과 나란히 비교

---


#### 5️⃣ 파라미터 비교 함수 : `adjust_parameters_and_test()`

##### ⚙️ 파라미터 비교 설정표 (adjust_parameters_and_test)

다양한 조건에서 신호등 인식 성능을 테스트할 수 있도록  
각 파라미터 세트를 미리 정의해 두었습니다.

| 테스트 이름 | `min_area` | `max_area` | `canny_low` | `canny_high` | `circularity_threshold` | 특징 |
|-------------|------------|------------|-------------|--------------|--------------------------|------|
| 기본값       | 100        | 5000       | 50          | 150          | 0.30                     | 평균적인 조건 |
| 더 민감하게   | 50         | 10000      | 30          | 120          | 0.20                     | 더 작은 객체도 탐지 |
| 더 엄격하게   | 200        | 8000       | 70          | 200          | 0.40                     | 정확도 우선, 작은 물체 배제 |
| 큰 신호등용   | 500        | 15000      | 40          | 160          | 0.25                     | 대형 신호등에 적합 |


#### 6️⃣ 단일 이미지 업로드 함수: `upload_and_detect()`

> 이 함수는 사용자가 단일 이미지를 업로드하면 **최적화된 파라미터**로 신호등을 자동 검출하고,  
> 그 결과를 시각적으로 보여주는 메인 실행 함수입니다.

---

##### 🧠 주요 기능 요약

| 기능 | 설명 |
|------|------|
| 이미지 업로드 | Google Colab의 `files.upload()` 이용 |
| 크기 조정 | 너비가 1200px 이상일 경우 리사이징 |
| 신호등 검출 | `detect_traffic_light_canny()` 함수 호출 |
| 시각화 출력 | `draw_detections()`로 결과 이미지 생성 후 `matplotlib` 시각화 |
| 정보 출력 | 신호등 개수, 위치, 크기 출력 |

---

##### ⚙️ 사용 파라미터 (최적화된 값)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `min_area` | 100 | 너무 작은 객체 제외 |
| `max_area` | 8000 | 너무 큰 객체 제외 |
| `canny_low` | 30 | Canny 경계 하한값 |
| `canny_high` | 120 | Canny 경계 상한값 |
| `circularity_threshold` | 0.25 | 원형성 기준 |
| `use_color_filter` | `True` | 색상 필터링 활성화 (빨강, 노랑, 초록, 파랑) |














