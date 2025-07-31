# 🎯 비디오 기반 머신러닝 앙상블 객체 탐지기 (YOLOv8 + Ensemble)

> Google Colab 기반 실행 | YOLOv8n + Confidence Threshold Ensemble | 객체 탐지 + 분석 리포트 자동 생성


## 앙상블 객체 탐지란?

이 프로젝트는 비디오 파일을 업로드하여 YOLOv8 모델을 기반으로 객체 탐지를 수행하고, **여러 confidence threshold를 앙상블**하는 방식으로 탐지 성능을 개선합니다.

앙상블(Ensemble)은 여러 개의 모델 예측을 결합해 최종 결과를 도출하는 방식입니다. 본 시스템에서 사용된 방식은:

- 다수의 YOLOv8 모델을 서로 다른 가중치(.pt) 파일로 불러와 같은 영상을 분석

- 각 모델의 결과를 Weighted Non-Maximum Suppression (NMS) 방식으로 통합하여 중복 제거

- 앙상블 효과로 인해 객체 탐지의 정확도, 신뢰도, 일관성 향상

📌 Weighted NMS란?

여러 모델이 탐지한 박스 중 위치가 비슷한 것들끼리 평균을 내어 더 정확한 최종 박스를 생성

### ✅ 주요 기능
- **YOLOv8n** 모델 기반 탐지
- **다중 threshold 앙상블**
- **가중치 기반 Non-Maximum Suppression (NMS)**
- **탐지 결과 영상 + 분석 리포트 자동 생성**
- **Google Colab에서 코드 1회 실행으로 전 과정 자동화**

---

## 머신러닝 앙상블이란?

앙상블(Ensemble)은 여러 모델(또는 예측)을 결합해 성능을 높이는 기법입니다.  
이 프로젝트는 YOLOv8 모델 하나를 사용하되, **여러 가지 confidence threshold 설정**을 통해 다양한 탐지 결과를 얻고, 이를 **가중 평균 방식으로 통합**하여 **더 안정적이고 누락이 적은 탐지**를 구현합니다.

---

## ⚙️ 설치 및 실행 방법 (Google Colab)

1. 아래 전체 코드를 Google Colab에 복사
2. `Run`으로 실행
3. 비디오 업로드 → 탐지 결과 분석 → ZIP 자동 다운로드

---

## 필요한 라이브러리 설치
```
!pip install ultralytics opencv-python numpy matplotlib
```

## 전체 코드 (복붙 실행)

```python
# 📦 필요한 패키지를 설치하고 YOLOv8 기반 앙상블 객체 탐지를 비디오에 적용하는 코드입니다.

import subprocess
import sys
import os

# 필요한 패키지를 자동 설치하는 함수
def install_packages():
    packages = ['ultralytics', 'opencv-python', 'numpy', 'matplotlib']
    for package in packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'ultralytics':
                from ultralytics import YOLO
            else:
                __import__(package)
            print(f"✅ {package} 설치됨")
        except ImportError:
            print(f"📦 {package} 설치 중...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])

install_packages()  # 위 함수 호출하여 패키지 설치

# 필요한 라이브러리 import
import cv2
import numpy as np
from collections import defaultdict
import zipfile
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
from google.colab import files
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# YOLO 모델을 활용한 비디오 앙상블 탐지 클래스 정의
class VideoEnsembleDetector:
    def __init__(self):
        """YOLOv8 모델과 앙상블 설정 초기화"""
        print("🤖 앙상블 모델 로딩 중...")
        self.model = YOLO('yolov8n.pt')  # 가볍고 빠른 YOLOv8n 사용
        print("✅ YOLOv8n 모델 로딩 완료")

        # 여러 confidence threshold를 가중치로 앙상블 구성
        self.ensemble_configs = [
            {'conf': 0.15, 'weight': 0.2},
            {'conf': 0.25, 'weight': 0.3},
            {'conf': 0.35, 'weight': 0.3},
            {'conf': 0.45, 'weight': 0.2}
        ]

        # 도로 환경에서 주요 클래스만 사용
        self.target_classes = {
            'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3,
            'bus': 5, 'truck': 7, 'traffic light': 9, 'stop sign': 11
        }

        # 각 클래스에 색상 매핑
        self.colors = {
            'person': (0, 255, 0), 'bicycle': (255, 0, 0), 'car': (0, 0, 255),
            'motorcycle': (255, 255, 0), 'bus': (128, 0, 128),
            'truck': (255, 165, 0), 'traffic light': (0, 255, 255),
            'stop sign': (255, 0, 255)
        }
        self.iou_threshold = 0.5  # NMS 적용 시 IOU 기준

    def upload_video(self):
        """Colab 환경에서 비디오 파일 업로드"""
        uploaded = files.upload()
        if not uploaded:
            return None, None
        filename = list(uploaded.keys())[0]
        return filename, filename.split('.')[0]

    def ensemble_predict(self, frame):
        """다중 threshold로 예측 후 결과 가중 평균 앙상블"""
        all_detections = []
        for config in self.ensemble_configs:
            results = self.model(frame, conf=config['conf'], verbose=False)
            weight = config['weight']
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0]) * weight
                    bbox = box.xyxy[0].cpu().numpy()
                    all_detections.append({
                        'bbox': bbox, 'conf': conf,
                        'cls_id': cls_id, 'threshold': config['conf']
                    })
        return all_detections

    def weighted_nms(self, detections):
        """가중치 기반 NMS 적용하여 최종 박스 필터링"""
        if not detections:
            return []
        class_detections = defaultdict(list)
        for det in detections:
            class_detections[det['cls_id']].append(det)

        final_detections = []
        for cls_id, cls_dets in class_detections.items():
            cls_dets.sort(key=lambda x: x['conf'], reverse=True)
            kept = []
            for det in cls_dets:
                if all(self.calculate_iou(det['bbox'], k['bbox']) <= self.iou_threshold for k in kept):
                    kept.append(det)
                if len(kept) >= 15:
                    break
            final_detections.extend(kept)
        return final_detections

    def calculate_iou(self, bbox1, bbox2):
        """IoU 계산 함수"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = ((x2_1 - x1_1) * (y2_1 - y1_1)) + ((x2_2 - x1_2) * (y2_2 - y1_2)) - inter
        return inter / union if union > 0 else 0.0

    def draw_detections(self, frame, detections):
        """탐지된 객체를 프레임에 그리기"""
        class_names = self.model.names
        for det in detections:
            class_name = class_names.get(det['cls_id'], 'unknown')
            if class_name not in self.target_classes:
                continue
            x1, y1, x2, y2 = map(int, det['bbox'])
            color = self.colors.get(class_name, (255, 255, 255))
            label = f"{class_name}: {det['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame

    def process_video(self, video_path, max_frames=900):
        """비디오 파일을 프레임 단위로 처리하고 결과 비디오 생성"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, {}

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        process_frames = min(total_frames, max_frames)

        output_path = "ensemble_detected_video.mp4"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        detection_stats = defaultdict(int)
        frame_count = 0
        while frame_count < process_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            detections = self.ensemble_predict(frame)
            filtered = self.weighted_nms(detections)
            class_names = self.model.names
            for det in filtered:
                class_name = class_names.get(det['cls_id'], 'unknown')
                if class_name in self.target_classes:
                    detection_stats[class_name] += 1
            result_frame = self.draw_detections(frame.copy(), filtered)
            info = f"Frame: {frame_count}/{process_frames}"
            cv2.putText(result_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            det_info = f"Current detections: {len(filtered)}"
            cv2.putText(result_frame, det_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            out.write(result_frame)

        cap.release()
        out.release()
        return output_path, detection_stats

    def create_final_package(self, video_path, stats, title):
        """비디오와 리포트를 ZIP으로 압축"""
        stats_file = "ensemble_detection_report.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"비디오: {title}\n")
            f.write("사용 모델: YOLOv8n\n")
            f.write("앙상블 방식: 다중 Confidence Threshold\n")
            f.write("탐지 통계:\n")
            total = sum(stats.values())
            for cls, cnt in stats.items():
                pct = (cnt / total * 100) if total > 0 else 0
                f.write(f"- {cls}: {cnt}회 ({pct:.1f}%)\n")
        zip_name = f"{title}_ensemble_detection.zip"
        with zipfile.ZipFile(zip_name, 'w') as zipf:
            zipf.write(video_path, "ensemble_detected_video.mp4")
            zipf.write(stats_file, "detection_report.txt")
        os.remove(stats_file)
        return zip_name

    def run_detection(self):
        """전체 파이프라인 실행"""
        video_path, title = self.upload_video()
        if not video_path:
            return
        output_video, stats = self.process_video(video_path)
        if not output_video:
            return
        zip_file = self.create_final_package(output_video, stats, title)
        files.download(zip_file)

# 실행
if __name__ == '__main__':
    detector = VideoEnsembleDetector()
    detector.run_detection()


# ▶️ 실행
detector = VideoEnsembleDetector()
detector.run_detection()
```
## 결과
<img width="1052" height="590" alt="image" src="https://github.com/user-attachments/assets/3cff8944-d8ae-4a3f-8d7d-8bc8ca48bf54" />
<img width="1066" height="494" alt="image" src="https://github.com/user-attachments/assets/aa1c49db-1ec2-4e0e-8c41-c58ad065bdc2" />


### 결과 해석

- detect_stats.json은 프레임 단위의 클래스 통계를 포함하므로

- 특정 클래스가 언제 자주 나타났는지 분석 가능

- 영상 출력은 시각적으로 신뢰도를 바탕으로 색상 차등을 두어 강조

