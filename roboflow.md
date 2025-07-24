# Roboflow를 활용한 이미지 라벨링, 작업 가이드

이 파일은 Roboflow를 활용하여 이미지 라벨링을 수행하고, YOLO 형식으로 데이터셋을 내보내는 과정을 단계별로 정리한 가이드입니다.

## ✅ 사전 준비

1. Roboflow 계정 생성 및 로그인:  
   https://roboflow.com 에서 계정 생성 및 로그인

2. 데이터셋 준비:  
   - 라벨링할 이미지(.jpg, .png,  등) 준비
   - 중복 이미지 제거 권장
   - 여러개의 영상을 업로드 해서 프레임 나누기

---

## 📁 프로젝트 생성

1. **[Create New Project]** 클릭
2. 프로젝트 이름 입력 (예: `0722_labeling`)
3. 프로젝트 타입 선택:
   - Project Type: `Object Detection`
   - Annotation Format: `YOLOv8` (or your preferred format)
4. [Create Project] 클릭

---

## ⬆️ 이미지 업로드

1. 상단 [Upload] 버튼 클릭
2. Drag & Drop으로 이미지 업로드
3. 중복 이미지가 있는 경우 자동으로 필터링됨
4. [Finish Uploading] 클릭 → [Annotate Now] 선택

---

## ✏️ 라벨링

1. 좌측 라벨 패널에서 원하는 클래스 생성 (ex: `traffic_light`, `lane`, `crosswalk`, `speed_sign` 등)
2. 단축키 사용 가능:
   - `B`: 박스 만들기 (bounding box)
   - `Delete`: 라벨 삭제
3. 클래스별 라벨링 가이드라인에 따라 정확하게 박스 지정
   - 예시:
     - `traffic_light`: 신호등 전체 영역
     - `lane`: 차선 (중앙선, 점선, 횡단보도 등)
     - `speed_sign`: 속도 제한 표지판
     - `lane_right_left`: 차선 방향 (좌/우회전 등)
4. 위 단계를 반복하여 여러장의 이미지 라벨랑 하기

✅ **팁:** [Settings] > [Label Assist] 기능을 통해 자동 라벨링 시도 가능

---

## 📦 라벨링 완료 후 내보내기 (Export)

1. 상단 메뉴 [Generate] 클릭
2. 원하는 버전명 설정 (예: `v1.0`)
3. Resize, Augmentation 여부 설정
4. [Generate] 클릭

---

## ⬇️ YOLO 형식으로 다운로드

1. 생성된 버전 클릭
2. [Download Dataset] 클릭
3. Format: `YOLOv8 PyTorch` 선택
4. [Download ZIP] 클릭 -> API Key 복사해두기

## 🔐 Roboflow API Key란?

**Roboflow API Key**는 사용자가 Roboflow 플랫폼에서 모델, 데이터셋, 프로젝트 등의 기능에 **프로그래밍 방식으로 접근**할 수 있도록 인증하는 **개인 고유 토큰**입니다.  
이를 통해 Python 코드 또는 외부 애플리케이션에서 Roboflow의 모델을 **불러오거나 예측 요청을 수행**할 수 있습니다.

> ✅ API Key는 비밀번호처럼 **외부에 노출되지 않도록 주의**해야 합니다.

---

## 🛠️ Roboflow API Key 생성 방법

1. Roboflow에 로그인  
   👉 https://roboflow.com

2. 우측 상단 프로필 아이콘 클릭  
   → `Settings` (또는 `Account`)

3. 좌측 메뉴에서 **"Roboflow API"** 선택

4. **"Create API Key"** 또는 기존 키 복사  
   → 생성된 키는 아래와 같이 생김:

---

압축 파일에는 다음이 포함됩니다:

### 📂 YOLOv8 데이터셋 폴더 구조 설명

Roboflow에서 YOLOv8 형식으로 내보낸 데이터셋은 다음과 같은 디렉토리 구조를 가집니다:



| 경로                  | 내용 설명 |
|-----------------------|-----------|
| `data.yaml`           | 클래스 정보 및 학습/검증 이미지 경로가 포함된 설정 파일 |
| `train/images/`       | 학습에 사용할 원본 이미지들 (.jpg, .png 등) |
| `train/labels/`       | 학습 이미지에 대한 라벨 정보 (.txt 파일, YOLO 형식) |
| `valid/images/`       | 검증(Validation)에 사용할 이미지들 |
| `valid/labels/`       | 검증 이미지의 라벨 정보 |


---

## ✅ 라벨링 시 주의 사항

- [x] 클래스명 오타 없이 통일
- [x] 박스는 너무 작거나 크지 않게 조정
- [x] 흐릿한 객체는 라벨링 X -> 나중에 객체 인식의 오류 원인이 됨
- [x] 라벨링 기준에 일관성 유지
- [x] 중복 객체는 모두 개별 라벨링

---
