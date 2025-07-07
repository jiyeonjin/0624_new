# 👁️‍🗨️ OpenCV: 컴퓨터 비전의 시작부터 끝까지 (Python 중심)

> OpenCV(Open Source Computer Vision Library)는 이미지 및 영상 처리, 컴퓨터 비전, 머신러닝, 딥러닝에 활용되는 세계적으로 가장 널리 사용되는 오픈소스 라이브러리입니다.

---

## 🧠 OpenCV란?

- **OpenCV**는 "Open Source Computer Vision Library"의 약자이며,
- 실시간 이미지 및 비디오 처리 기능을 제공합니다.
- C++, Python, Java 등을 지원하며, 특히 Python 버전은 `opencv-python` 패키지로 제공됩니다.

---

## 📘 OpenCV에서 자주 등장하는 용어들

| 용어 | 설명 |
|------|------|
| **BGR** | OpenCV의 기본 색상 순서 (Blue, Green, Red) |
| **Grayscale** | 흑백 이미지 (단일 채널, 밝기 정보만 포함) |
| **Thresholding** | 밝기 기준으로 픽셀을 흑백으로 나누는 처리 |
| **Edge Detection** | 경계를 검출하는 기법 (예: Canny) |
| **Contours** | 이미지 내 객체 외곽선 |
| **ROI (Region of Interest)** | 관심 영역: 분석/처리에 집중할 이미지 영역 |
| **Kernel (필터)** | 영상 처리 시 사용하는 행렬 마스크 |
| **Hough Transform** | 직선, 원 등을 수학적으로 검출하는 알고리즘 |
| **Morphology** | 침식(Erosion), 팽창(Dilation) 등의 구조적 이미지 처리 |

---

## 📚 사용하는 주요 라이브러리

| 라이브러리 | 설명 |
|------------|------|
| `opencv-python` (`cv2`) | OpenCV의 Python 바인딩 |
| `numpy` | 이미지 = Numpy 배열, 수치 계산 필수 |
| `matplotlib` | 이미지 시각화 및 디버깅 |
| `Pillow` | 이미지 저장/포맷 변환에 사용 (선택) |

```bash
pip install opencv-python numpy matplotlib pillow
