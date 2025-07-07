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

## Open CV 사용한 예제 코드
```bash
# OpenCV 교통표지판 인식 (훈련 불필요!)


!pip install opencv-python matplotlib pillow
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import io
from PIL import Image

# 한글 폰트 설정 (Colab용)
import matplotlib.font_manager as fm

# Colab에서 한글 폰트 설치
!apt-get install -y fonts-nanum
!fc-cache -fv
!rm ~/.cache/matplotlib -rf

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

class OpenCVTrafficSignRecognizer:
    def __init__(self):
        # 색상 범위 정의 (HSV 색공간)
        self.color_ranges = {
            'red': {
                'lower1': np.array([0, 120, 70]),    # 빨간색 범위1
                'upper1': np.array([10, 255, 255]),
                'lower2': np.array([170, 120, 70]),  # 빨간색 범위2
                'upper2': np.array([180, 255, 255]),
                'sign_type': 'stop_prohibition'
            },
            'blue': {
                'lower': np.array([90, 100, 50]),    # 파란색
                'upper': np.array([150, 255, 255]),
                'sign_type': 'direction_guide'
            },
            'yellow': {
                'lower': np.array([15, 150, 150]),   # 노란색
                'upper': np.array([35, 255, 255]),
                'sign_type': 'warning_caution'
            },
            'green': {
                'lower': np.array([40, 150, 100]),   # 초록색
                'upper': np.array([80, 255, 255]),
                'sign_type': 'safety_permission'
            }
        }

    def detect_color(self, image):
        """색상 기반 표지판 분류"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        detected_colors = []

        for color_name, color_info in self.color_ranges.items():
            if color_name == 'red':
                # 빨간색은 HSV에서 두 범위로 나뉨
                mask1 = cv2.inRange(hsv, color_info['lower1'], color_info['upper1'])
                mask2 = cv2.inRange(hsv, color_info['lower2'], color_info['upper2'])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, color_info['lower'], color_info['upper'])

            # 색상 픽셀 수 계산
            color_pixels = cv2.countNonZero(mask)
            total_pixels = image.shape[0] * image.shape[1]
            color_ratio = color_pixels / total_pixels

            if color_ratio > 0.1:  # 10% 이상이면 해당 색상으로 판정
                detected_colors.append({
                    'color': color_name,
                    'ratio': color_ratio,
                    'sign_type': color_info['sign_type'],
                    'mask': mask
                })

        # 가장 많은 색상 반환
        if detected_colors:
            return max(detected_colors, key=lambda x: x['ratio'])
        return None

    def detect_shapes(self, image):
        """모양 기반 표지판 분류"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 에지 검출
        edges = cv2.Canny(blurred, 50, 150)

        # 컨투어 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shapes = []

        for contour in contours:
            # 작은 컨투어 무시
            if cv2.contourArea(contour) < 500:
                continue

            # 컨투어 근사화
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 꼭짓점 수로 모양 판별
            vertices = len(approx)

            # 모양 분류
            if vertices == 3:
                shape_type = "triangle_warning_sign"
            elif vertices == 4:
                # 사각형인지 확인
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 0.95 <= aspect_ratio <= 1.05:
                    shape_type = "square_general_sign"
                else:
                    shape_type = "rectangle_guide_sign"
            elif 5 <= vertices <= 10:
                shape_type = "octagon_stop_sign"
            else:
                # 원형 여부 확인
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:
                        shape_type = "circle_direction_sign"
                    else:
                        shape_type = "other_shape"
                else:
                    shape_type = "other_shape"

            shapes.append({
                'shape': shape_type,
                'vertices': vertices,
                'area': cv2.contourArea(contour),
                'contour': contour
            })

        # 가장 큰 모양 반환
        if shapes:
            return max(shapes, key=lambda x: x['area'])
        return None

    def detect_text_patterns(self, image):
        """텍스트 패턴 인식 (간단한 템플릿 매칭)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 이진화
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # STOP 텍스트 특징 검사 (간단한 방법)
        # 중앙 영역의 백색 픽셀 패턴 확인
        h, w = binary.shape
        center_region = binary[h//3:2*h//3, w//4:3*w//4]
        white_pixels = cv2.countNonZero(center_region)
        total_pixels = center_region.shape[0] * center_region.shape[1]

        text_patterns = []

        if white_pixels / total_pixels > 0.3:  # 30% 이상 흰 픽셀
            text_patterns.append("text_included_stop_possible")

        return text_patterns

    def classify_traffic_sign(self, image):
        """종합적인 교통표지판 분류"""
        # 1. 색상 검출
        color_result = self.detect_color(image)

        # 2. 모양 검출
        shape_result = self.detect_shapes(image)

        # 3. 텍스트 패턴 검출
        text_result = self.detect_text_patterns(image)

        # 4. 종합 판단
        classification = {
            'color_info': color_result,
            'shape_info': shape_result,
            'text_info': text_result,
            'final_prediction': 'unknown_sign'
        }

        # 규칙 기반 분류
        if color_result and shape_result:
            color = color_result['color']
            shape = shape_result['shape']

            if color == 'red':
                if 'octagon' in shape or 'text_included' in str(text_result):
                    classification['final_prediction'] = 'stop_sign'
                else:
                    classification['final_prediction'] = 'prohibition_sign'
            elif color == 'blue':
                if 'circle' in shape:
                    classification['final_prediction'] = 'direction_sign'
                else:
                    classification['final_prediction'] = 'guide_sign'
            elif color == 'yellow':
                if 'triangle' in shape:
                    classification['final_prediction'] = 'warning_sign'
                else:
                    classification['final_prediction'] = 'caution_sign'
            elif color == 'green':
                classification['final_prediction'] = 'safety_sign'

        elif color_result:
            # 색상만으로 판단
            if color_result['sign_type'] == 'stop_prohibition':
                classification['final_prediction'] = 'stop_sign'
            elif color_result['sign_type'] == 'direction_guide':
                classification['final_prediction'] = 'direction_sign'
            elif color_result['sign_type'] == 'warning_caution':
                classification['final_prediction'] = 'warning_sign'
            elif color_result['sign_type'] == 'safety_permission':
                classification['final_prediction'] = 'safety_sign'
            else:
                classification['final_prediction'] = color_result['sign_type']

        elif shape_result:
            # 모양만으로 판단
            if 'octagon' in shape_result['shape']:
                classification['final_prediction'] = 'stop_sign_by_shape'
            elif 'triangle' in shape_result['shape']:
                classification['final_prediction'] = 'warning_sign_by_shape'
            elif 'circle' in shape_result['shape']:
                classification['final_prediction'] = 'direction_sign_by_shape'

        return classification

    def visualize_detection(self, image, classification):
        """검출 결과 시각화"""
        plt.figure(figsize=(15, 10))

        # 1. 원본 이미지
        plt.subplot(2, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        # 2. HSV 이미지
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        plt.subplot(2, 3, 2)
        plt.imshow(hsv)
        plt.title('HSV Color Space')
        plt.axis('off')

        # 3. 색상 마스크
        plt.subplot(2, 3, 3)
        if classification['color_info']:
            plt.imshow(classification['color_info']['mask'], cmap='gray')
            plt.title(f"Color Detection: {classification['color_info']['color']}")
        else:
            plt.text(0.5, 0.5, 'Color Not Detected', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Color Detection Failed')
        plt.axis('off')

        # 4. 에지 검출
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        plt.subplot(2, 3, 4)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detection')
        plt.axis('off')

        # 5. 모양 검출
        plt.subplot(2, 3, 5)
        shape_image = image.copy()
        if classification['shape_info']:
            cv2.drawContours(shape_image, [classification['shape_info']['contour']], -1, (255, 0, 0), 3)
            plt.imshow(shape_image)
            plt.title(f"Shape: {classification['shape_info']['shape']}")
        else:
            plt.imshow(image)
            plt.title('Shape Not Detected')
        plt.axis('off')

        # 6. 최종 결과
        plt.subplot(2, 3, 6)
        plt.text(0.5, 0.7, '🎯 Final Prediction', ha='center', va='center',
                transform=plt.gca().transAxes, fontsize=16, fontweight='bold')

        # 영어 예측 결과를 읽기 쉬운 영어로 변환해서 표시
        prediction_english = self.convert_prediction_to_english(classification['final_prediction'])
        plt.text(0.5, 0.4, prediction_english, ha='center', va='center',
                transform=plt.gca().transAxes, fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        # 상세 정보 (영어로)
        details = ""
        if classification['color_info']:
            details += f"Color: {classification['color_info']['color']}\n"
        if classification['shape_info']:
            details += f"Shape: {classification['shape_info']['shape']}\n"
        if classification['text_info']:
            details += f"Text: {classification['text_info']}"

        plt.text(0.5, 0.1, details, ha='center', va='center',
                transform=plt.gca().transAxes, fontsize=10)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def convert_prediction_to_english(self, prediction):
        """영어 예측 결과를 더 읽기 쉬운 영어로 변환"""
        conversion_dict = {
            'stop_sign': 'STOP Sign',
            'direction_sign': 'Direction Sign',
            'warning_sign': 'Warning Sign',
            'caution_sign': 'Caution Sign',
            'safety_sign': 'Safety Sign',
            'prohibition_sign': 'Prohibition Sign',
            'guide_sign': 'Guide Sign',
            'stop_sign_by_shape': 'STOP Sign (by shape)',
            'warning_sign_by_shape': 'Warning Sign (by shape)',
            'direction_sign_by_shape': 'Direction Sign (by shape)',
            'unknown_sign': 'Unknown Sign',
            'stop_prohibition': 'STOP/Prohibition Sign',
            'direction_guide': 'Direction/Guide Sign',
            'warning_caution': 'Warning/Caution Sign',
            'safety_permission': 'Safety/Permission Sign'
        }

        result = conversion_dict.get(prediction, f'Unclassified: {prediction}')
        print(f"🔍 Debug: '{prediction}' → '{result}'")
        return result

    def upload_and_analyze(self):
        """이미지 업로드 및 분석"""
        print("📷 교통표지판 이미지를 업로드해주세요!")
        print("=" * 50)

        uploaded = files.upload()

        if not uploaded:
            print("❌ 업로드된 파일이 없습니다.")
            return

        for filename, file_data in uploaded.items():
            print(f"\n🔍 '{filename}' 분석 중...")

            try:
                # 이미지 로드
                image = Image.open(io.BytesIO(file_data))
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                image_array = np.array(image)

                # 분석 수행
                result = self.classify_traffic_sign(image_array)

                # 결과 출력
                print(f"\n🎯 분석 결과: {result['final_prediction']}")

                if result['color_info']:
                    print(f"🎨 검출된 색상: {result['color_info']['color']} ({result['color_info']['ratio']:.1%})")

                if result['shape_info']:
                    print(f"📐 검출된 모양: {result['shape_info']['shape']}")

                if result['text_info']:
                    print(f"📝 텍스트 정보: {result['text_info']}")

                # 시각화
                self.visualize_detection(image_array, result)

                print("-" * 50)

            except Exception as e:
                print(f"❌ 오류 발생: {e}")

def main():
    """메인 실행 함수"""
    print("🚗 OpenCV 교통표지판 인식 시스템")
    print("=" * 50)
    print("🎯 특징: 훈련 불필요, 즉시 인식!")
    print("🔍 방법: 색상 + 모양 + 텍스트 패턴 분석")
    print("📊 인식 가능: 정지, 지시, 경고, 안내 표지판")
    print("=" * 50)

    # 시스템 초기화
    recognizer = OpenCVTrafficSignRecognizer()

    # 이미지 분석
    while True:
        recognizer.upload_and_analyze()

        continue_choice = input("\n🔄 다른 이미지도 분석하시겠어요? (y/n): ")
        if continue_choice.lower() != 'y':
            break

    print("\n🎉 OpenCV 교통표지판 인식 완료!")

if __name__ == "__main__":
    main()
```

# 🚦 OpenCV 기반 교통표지판 인식 시스템 (비학습 기반)

> ✅ 위의 코드는 머신러닝 학습 없이! 색상 + 모양 + 텍스트 패턴을 기반으로 교통 표지판을 실시간으로 인식하는 Python + OpenCV 시스템입니다.

---

## 📌 프로젝트 개요

- **라이브러리**: `opencv-python`, `matplotlib`, `Pillow`, `numpy`, `google.colab`
- **기술 특징**:
  - **딥러닝 훈련 불필요** (즉시 실행)
  - HSV 색상 공간 기반 **색상 인식**
  - 컨투어 기반 **모양 분석**
  - 텍스트 포함 여부 판단 **패턴 검출**
- **인식 대상**: `정지`, `지시`, `경고`, `안내`, `금지`, `안전` 표지판

---

## 🧱 예제 코드 구조 분석 요약

### 🔸 클래스: `OpenCVTrafficSignRecognizer`

| 구성 요소 | 설명 |
|-----------|------|
| `__init__()` | HSV 색상 범위 초기화 |
| `detect_color()` | 이미지 색상 영역 분석 후 색상 종류 및 비율 반환 |
| `detect_shapes()` | 컨투어 기반 모양 분석 (삼각형, 사각형, 원, 팔각형 등) |
| `detect_text_patterns()` | 중심 흰 픽셀 비율로 텍스트 유무 추정 |
| `classify_traffic_sign()` | 색상 + 모양 + 텍스트 정보 종합 → 표지판 분류 |
| `visualize_detection()` | 원본 이미지 + 마스크 + 에지 + 모양 시각화 |
| `convert_prediction_to_english()` | 예측 결과를 직관적인 영어 표현으로 변환 |
| `upload_and_analyze()` | 이미지 업로드 → 분석 → 시각화 |
| `main()` | 사용자 인터페이스 실행 루프 |

---

## 🧠 핵심 인식 방식 요약

### 🎨 색상 인식 (HSV 기반)

| 색상 | HSV 범위 | 의미 |
|------|----------|------|
| 빨강 (`red`) | [0~10], [170~180] | 정지, 금지 표지 |
| 파랑 (`blue`) | [90~150] | 지시, 안내 표지 |
| 노랑 (`yellow`) | [15~35] | 경고 표지 |
| 초록 (`green`) | [40~80] | 안전, 허용 표지 |

```python
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
mask = cv2.inRange(hsv, lower, upper)
ratio = countNonZero(mask) / total_pixels

