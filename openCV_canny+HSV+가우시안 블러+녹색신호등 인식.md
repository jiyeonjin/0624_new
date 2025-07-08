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
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image
import io
```
