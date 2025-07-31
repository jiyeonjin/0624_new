# 🎯 비디오 기반 머신러닝 앙상블 객체 탐지기 (YOLOv8 + Ensemble)

> Google Colab 기반 실행 | YOLOv8n + Confidence Threshold Ensemble | 객체 탐지 + 분석 리포트 자동 생성

---

## 앙상블 개요

이 프로젝트는 비디오 파일을 업로드하여 YOLOv8 모델을 기반으로 객체 탐지를 수행하고, **여러 confidence threshold를 앙상블**하는 방식으로 탐지 성능을 개선합니다.

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

## 전체 코드 (복붙 실행)

```python
# 📦 패키지 설치 함수
import subprocess
import sys
import os

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

install_packages()

# 🧠 라이브러리 불러오기
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

print("✅ 모든 라이브러리 로딩 완료!")

# 🤖 객체 탐지 클래스 정의
class VideoEnsembleDetector:
    def __init__(self):
        print("🤖 앙상블 모델 로딩 중...")
        try:
            self.model = YOLO('yolov8n.pt')
            print("✅ YOLOv8n 모델 로딩 완료")
            self.ensemble_configs = [
                {'conf': 0.15, 'weight': 0.2},
                {'conf': 0.25, 'weight': 0.3},
                {'conf': 0.35, 'weight': 0.3},
                {'conf': 0.45, 'weight': 0.2}
            ]
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            return

        self.target_classes = {
            'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3,
            'bus': 5, 'truck': 7, 'traffic light': 9, 'stop sign': 11
        }
        self.colors = {
            'person': (0, 255, 0), 'bicycle': (255, 0, 0), 'car': (0, 0, 255),
            'motorcycle': (255, 255, 0), 'bus': (128, 0, 128),
            'truck': (255, 165, 0), 'traffic light': (0, 255, 255),
            'stop sign': (255, 0, 255)
        }
        self.iou_threshold = 0.5
        print("🎯 앙상블 탐지기 초기화 완료!")

    def upload_video(self):
        print("📁 비디오 파일을 선택해주세요...")
        uploaded = files.upload()
        if not uploaded:
            return None, None
        filename = list(uploaded.keys())[0]
        return filename, filename.split('.')[0]

    def ensemble_predict(self, frame):
        all_detections = []
        for config in self.ensemble_configs:
            try:
                results = self.model(frame, conf=config['conf'], verbose=False)
                weight = config['weight']
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0]) * weight
                            bbox = box.xyxy[0].cpu().numpy()
                            all_detections.append({
                                'bbox': bbox,
                                'conf': conf,
                                'cls_id': cls_id,
                                'threshold': config['conf']
                            })
            except Exception:
                continue
        return all_detections

    def weighted_nms(self, detections):
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
                bbox1 = det['bbox']
                if all(self.calculate_iou(bbox1, k['bbox']) <= self.iou_threshold for k in kept):
                    kept.append(det)
                    if len(kept) >= 15:
                        break
            final_detections.extend(kept)
        return final_detections

    def calculate_iou(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter
        return inter / union if union > 0 else 0.0

    def draw_detections(self, frame, detections):
        class_names = self.model.names
        for det in detections:
            bbox, conf, cls_id = det['bbox'], det['conf'], det['cls_id']
            class_name = class_names.get(cls_id, 'unknown')
            if class_name not in self.target_classes:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            color = self.colors.get(class_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame

    def process_video(self, video_path, max_frames=900):
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
        frame_count = 0
        detection_stats = defaultdict(int)
        while frame_count < process_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            detections = self.ensemble_predict(frame)
            filtered = self.weighted_nms(detections)
            for det in filtered:
                class_name = self.model.names.get(det['cls_id'], 'unknown')
                if class_name in self.target_classes:
                    detection_stats[class_name] += 1
            result_frame = self.draw_detections(frame.copy(), filtered)
            out.write(result_frame)
        cap.release()
        out.release()
        return output_path, detection_stats

    def create_final_package(self, video_path, stats, title):
        stats_file = "ensemble_detection_report.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("앙상블 객체 탐지 결과 리포트\n\n")
            f.write(f"비디오 파일: {title}\n")
            f.write(f"사용 모델: YOLOv8n\n")
            f.write(f"Threshold: {[c['conf'] for c in self.ensemble_configs]}\n")
            f.write("\n탐지 결과:\n")
            total = sum(stats.values())
            for k, v in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                pct = (v / total * 100) if total > 0 else 0
                f.write(f"{k:15}: {v:4d}회 ({pct:5.1f}%)\n")
        zip_filename = f"{title}_ensemble_detection.zip"
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if os.path.exists(video_path):
                zipf.write(video_path, "ensemble_detected_video.mp4")
            zipf.write(stats_file, "detection_report.txt")
        os.remove(stats_file)
        return zip_filename

    def run_detection(self):
        print("🎯 비디오 앙상블 탐지 시작!")
        video_path, title = self.upload_video()
        if not video_path:
            return
        output_video, stats = self.process_video(video_path)
        if not output_video:
            return
        print(f"📊 총 탐지 횟수: {sum(stats.values())}회")
        zip_file = self.create_final_package(output_video, stats, title)
        print(f"📦 결과 패키지: {zip_file}")
        files.download(zip_file)

# ▶️ 실행
detector = VideoEnsembleDetector()
detector.run_detection()
```
## 결과
<img width="1052" height="590" alt="image" src="https://github.com/user-attachments/assets/3cff8944-d8ae-4a3f-8d7d-8bc8ca48bf54" />
<img width="1066" height="494" alt="image" src="https://github.com/user-attachments/assets/aa1c49db-1ec2-4e0e-8c41-c58ad065bdc2" />

