# 차선 인식 프로젝트 (SegFormerForSemanticSegmentation + 전이학습)
**팀원:** 윤은식, 전은서, 박현욱, 유성일, 지연진

---

## 📋 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [프로젝트를 위한 데이터 준비하기](#2-프로젝트를-위한-데이터-준비하기)
3. [단계별 진행 가이드](#3-단계별-진행-가이드)
4. [트러블슈팅](#4-트러블슈팅)
5. [다운로드 한 데이터셋을 가지고 코드 실행하기](#5-다운로드-한-데이터셋을-가지고-코드-실행하기)

---

## 1. 프로젝트 개요

### 프로젝트 목표
- 이 프로젝트는 **Hugging Face & NVIDIA 협업 SegFormerForSemanticSegmentation 모델**을 사용하여 **차선 인식(Lane Detection)**을 수행합니다.
- `seg11xl.pt` 사전 학습 가중치를 활용해 **전이학습(Transfer Learning)**으로 차선 픽셀 분류 모델을 학습합니다.

### 🛠 기술 스택
- **모델:** SegFormerForSemanticSegmentation (`seg11xl.pt` 기반)
- **데이터 라벨링:** Roboflow (Semantic Segmentation)
- **프로그래밍:** Python, PyTorch
- **환경:** Google Colab, RunPod

---

## 2. 프로젝트를 위한 데이터 준비하기

### ✅ 데이터 준비 (Roboflow)

#### 프로젝트 생성
1. Roboflow 접속 → `Create New Project`
2. **Project Type:** *Semantic Segmentation*
3. 프로젝트 이름: `lane-detection` (자유롭게 설정 가능)
4. 교수님께서 주신 영상 합쳐 업로드 (22분 가량)

### ⚠️ 데이터 준비 전 핵심 주의 사항

#### 잘못된 접근법
- **Object Detection** 프로젝트 타입 선택
- 결과: Image and Annotation Format에서 **semantic segmentation masks 옵션이 없음**

#### 올바른 접근법
- **Instance Segmentation** 프로젝트 타입 선택
- 결과: segmentation masks 옵션 제공으로 원하는 데이터 형식 획득 가능
- <img width="600" height="468" alt="image" src="https://github.com/user-attachments/assets/97a9b092-91df-47c6-9afd-9959dfc0028d" />
위 이미지와 같이 선택하여 프로젝트 생성하기.

---

## 3. 단계별 진행 가이드

### 🛠️ 1단계: Roboflow 프로젝트 생성
1. Roboflow 플랫폼 접속
2. 새 프로젝트 생성 시 **반드시 "Instance Segmentation" 선택**
   - ⚠️ Object Detection 선택 시 semantic segmentation masks 옵션 부재
3. 프로젝트 이름 및 기본 설정 완료

### 🛠️ 2단계: 데이터 업로드 및 라벨링
1. 차선 이미지 데이터 업로드
2. Segmentation 방식으로 차선 영역 라벨링
   - 픽셀 단위로 정확한 차선 경계 표시
   - 다양한 차선 유형 고려 (실선, 점선, 중앙선 등)

### 🛠️ 3단계: 클래스 정의

#### 클래스 설정 가이드
> ⚠️ **처음에는 단일 클래스 추천** → 데이터 수가 충분해지면 세부 클래스 추가 가능  
> 우리팀의 경우 모든 차선을 'lane' 하나의 단일 클래스로만 간주

| 클래스명       | 설명                                    |
|----------------|----------------------------------------|
| `lane`         | 모든 차선 (색상/형태 관계없이)          |
| `lane_white`   | 흰색 차선 (선택사항)                    |
| `lane_yellow`  | 노란색 차선 (선택사항)                  |
| `lane_dashed`  | 점선 차선 (선택사항)                    |
| `lane_solid`   | 실선 차선 (선택사항)                    |

### 🛠️ 4단계: 라벨링 규칙 및 주의사항

#### 라벨링 규칙
차선 픽셀을 정확하게 구분하는 것이 목표입니다. 폴리건으로 진행하여 정확히 차선만 라벨링 하였습니다.
팀원분들은 다음 규칙을 따라 라벨링 해주세요.

##### 기본 규칙
1. **차선 전체 폭 라벨링**  
   - 중심선만 그리지 말고 실제 보이는 차선 두께 그대로 마스크 처리
2. **보이는 부분만 라벨링**  
   - 차량/사물에 가려진 부분은 추정하지 말고 보이는 영역만 칠하기
3. **클래스에 맞게 구분**  
   - 단일 클래스(`lane`)만 쓰는 경우 색상, 형태 구분 없이 모두 같은 클래스에 라벨링
4. **정확한 경계**  
   - 도로와 차선의 경계가 헷갈리는 경우 확대하여 픽셀 단위로 정밀하게
5. **배경 포함 금지**  
   - 도로, 차선 외의 영역(차량, 보도, 하늘 등)은 절대 라벨링하지 않음

##### ⚠️ 주의사항
- 동일 장면에서 연속 프레임은 과도하게 포함하지 말 것 (데이터 중복 방지)
- 다양한 조건(맑음, 비, 야간, 역광, 그림자 포함)으로 데이터 확보
- 곡선 차선, 교차로 차선, 다차선 도로 등 다양한 형태 반영

##### 데이터 Export 권장 설정
- **Export Format:** COCO Segmentation
- **Images:** JPG/PNG
- **Masks:** PNG (클래스별 색상 구분)
- **Train/Valid/Test Split:** 70% / 20% / 10% 추천 

### 🛠️ 5단계: 데이터셋 다운로드
1. **Image and Annotation Format**에서 **"semantic segmentation masks"** 선택
 <img width="900" height="700" alt="image" src="https://github.com/user-attachments/assets/56d8e0a8-1dd7-4be8-a49f-aebd8baa73f4" />

2. 원하는 형식으로 데이터셋 export
3. 로컬 환경으로 computer to zip 다운로드

---

## 4. 트러블슈팅

### 🔍 주요 문제 해결 방법

#### 문제: Semantic Segmentation Masks 옵션이 보이지 않음
**원인:** Object Detection 프로젝트 타입으로 생성  
**해결책:** 프로젝트를 Instance Segmentation으로 새로 생성

#### 문제: 라벨링 품질 저하
**해결책:** 
- 충분한 데이터 다양성 확보
- 정확한 픽셀 단위 라벨링 수행
- 다양한 환경 조건의 이미지 포함

---

## 5. 다운로드 한 데이터셋을 가지고 코드 실행하기 (Colab, RunPods)

### 📝 코랩 환경에서 실행하기
[차선 인식 프로젝트 코랩 노트북](https://colab.research.google.com/github/jiyeonjin/0624_new/blob/main/0813_%ED%8C%80%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8.ipynb)

위 링크를 통해 전체 구현 코드와 실행 결과를 확인할 수 있습니다.


### 📹 프로젝트 결과 영상
**차선 인식 모델 실행 결과 데모 영상 (30초)**

![차선 인식 데모](https://github.com/jiyeonjin/0624_new/raw/main/assets/demo.gif.gif)

<p align="center">
  <img src="https://img.shields.io/badge/🎬_차선_인식_결과-실시간_데모-4CAF50?style=for-the-badge&logo=videocam&logoColor=white" alt="차선 인식 결과"/>
  <br>
  <sub>📊 SegFormerForSemanticSegmentation 모델 추론 결과</sub>
</p>

---

### 코랩에서 실행한 코드 상세 분석

코랩 차선 인식 프로젝트 코드는 다음과 같은 순서로 진행됩니다:

```
1. 데이터 정리 → 2. 환경 설정 → 3. 데이터 준비 → 4. 모델 로딩 → 5. 학습 → 6. 추론 → 7. 결과 확인
```

### 주요 구성 요소
- **데이터셋**: Roboflow에서 다운로드한 라벨링된 이미지들
- **모델**: SegFormerForSemanticSegmentation (사전 훈련된 모델)
- **학습 코드**: 전이학습을 위한 파인튜닝 코드
- **추론 코드**: 새로운 이미지에서 차선을 찾는 코드


#### ✅ 실제 프로젝트 코드 분석 1

```python
import os
import shutil

def separate_images_and_masks(data_dir):
    # 이미지와 마스크 파일을 구분하는 확장자 또는 규칙에 맞게 분류
    image_exts = ['.jpg', '.jpeg', '.png']  # 실제 이미지 확장자
    mask_exts = ['.png']                    # 마스크 확장자 (보통 png)
    
    # 새 폴더 경로 지정
    image_folder = os.path.join(data_dir, 'images')
    mask_folder = os.path.join(data_dir, 'masks')
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    
    # 데이터 폴더 내 파일 목록 가져오기
    all_files = os.listdir(data_dir)
    
    for file_name in all_files:
        file_path = os.path.join(data_dir, file_name)
        
        # 파일 여부 확인
        if os.path.isfile(file_path):
            ext = os.path.splitext(file_name)[1].lower()
            
            # 확장자에 따라 폴더로 이동
            if ext in image_exts and 'mask' not in file_name.lower():
                shutil.move(file_path, os.path.join(image_folder, file_name))
            elif ext in mask_exts and 'mask' in file_name.lower():
                shutil.move(file_path, os.path.join(mask_folder, file_name))
    
    print(f"{data_dir} 내 이미지와 마스크 파일을 분리해 각각 images/, masks/ 폴더에 옮겼습니다.")

# train, valid, test 각각에 대해 실행
base_dir = '/content/data'
for split in ['train', 'valid', 'test']:
    split_dir = os.path.join(base_dir, split)
    separate_images_and_masks(split_dir)
```

**👉 실행 결과**:
```
/content/data/train/images/    ← 훈련용 이미지들
/content/data/train/masks/     ← 훈련용 마스크들
/content/data/valid/images/    ← 검증용 이미지들  
/content/data/valid/masks/     ← 검증용 마스크들
/content/data/test/images/     ← 테스트용 이미지들
/content/data/test/masks/      ← 테스트용 마스크들
```

### 이 코드가 중요한 이유

**딥러닝 모델 학습을 위해서는 데이터가 다음과 같이 정리되어야 합니다**:
- 이미지와 마스크가 별도 폴더에 정리
- 훈련/검증/테스트 세트로 구분
- 일관된 폴더 구조 유지

이 코드는 **데이터 전처리의 첫 번째 단계**로, 이후 모든 학습 과정의 기반이 됩니다!

#### ✅ 실제 프로젝트 코드 분석 2


```python
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LaneSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir         # 원본 이미지 폴더 경로
        self.mask_dir = mask_dir           # 마스크 이미지 폴더 경로
        self.transform = transform         # 데이터 증강 설정
        self.images = sorted(os.listdir(image_dir))   # 이미지 파일 리스트
        self.masks = sorted(os.listdir(mask_dir))     # 마스크 파일 리스트
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 파일 경로 생성
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        # 이미지 불러오기 (BGR → RGB 변환)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 마스크 불러오기 (그레이스케일)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 크기가 다르면 마스크를 이미지 크기에 맞춤
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # 마스크 이진화: 0이 아닌 모든 값을 차선(1)으로 변환
        mask = (mask != 0).astype('float32')
        
        # 데이터 증강 적용
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 마스크에 채널 차원 추가 후 반환
        return image, mask.unsqueeze(0)
```

#### 코드 상세 분석 (주요 기능 설명)
- 차선 분할(Lane Segmentation) AI 모델 학습을 위한 PyTorch 데이터셋 클래스입니다.
- 🖼️ **이미지 & 마스크 로드**: 원본 도로 이미지와 차선 마스크를 자동으로 불러옴
- 🔄 **자동 크기 조정**: 이미지와 마스크 크기가 다를 때 자동으로 맞춤
- 🎯 **이진화 처리**: 차선(1) vs 배경(0)으로 단순화
- 🔀 **데이터 증강**: Albumentations 라이브러리 지원

#### ✅ 실제 프로젝트 코드 분석 3

```python
import matplotlib.pyplot as plt
import cv2
import os

# 마스크 이미지 경로 설정 (실제 파일명으로 변경)
mask_path = '/content/data/train/masks/-02_mp4-0035_jpg.rf.1d0c62696d772f20003e3d3476502c93_mask.png'

# 마스크 이미지 불러오기 (그레이스케일)
mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 마스크 이미지 시각화
plt.imshow(mask_img, cmap='gray')
plt.title("Sample Mask Image")
plt.show()

# 마스크에 포함된 고유한 픽셀 값들 확인
print("Unique values in mask:", set(mask_img.flatten()))
```

#### 위 코드 출력 결과
> <img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/3980779c-18ba-48ab-a38d-b240916e05aa" />


#### 코드 상세 분석 (주요 기능 설명)
- 차선 분할 학습 전에 마스크 이미지가 올바르게 라벨링되어 있는지 확인하는 코드입니다.
- 🔍 **마스크 시각화**: 마스크 이미지를 그레이스케일로 표시
- 📊 **픽셀값 확인**: 마스크에 포함된 고유한 픽셀 값들을 출력
- 🖥 **라벨링 검증**: 차선과 배경이 제대로 구분되어 있는지 확인

#### 올바른 마스크
- **배경**: 픽셀값 0 (검은색)
- **차선**: 픽셀값 255 또는 0이 아닌 값 (흰색/회색)
- **출력 예시**: `{0, 255}` 또는 `{0, 38, 76, 113, 150, 255}`

#### ❌ 문제가 있는 경우
- 모든 픽셀이 같은 값 (예: `{0}` 또는 `{255}`)
- 마스크가 로드되지 않음 (파일 경로 확인 필요)

### 트러블슈팅 과정 (데이터셋 파일 자동 분류가 되는 문제)

### 문제 상황
로보플로우에서 다운받은 데이터셋이 아래와 같은 구조로 되어 있어 모델이 인식하지 못했습니다:

```
train/
├── image1.jpg
├── image1_mask.png  
├── image2.jpg
├── image2_mask.png
└── ...
```

**문제점**: 이미지와 마스크가 같은 폴더에 섞여있음 → 데이터로더 오류 발생!

### 해결 방법

### 자동 분류 규칙
- **이미지 파일**: `.jpg` 확장자 → `images/` 폴더로 이동
- **마스크 파일**: `.png` 확장자 + 파일명에 `'mask'` 포함 → `masks/` 폴더로 이동
- **일괄 처리**: train, valid, test 폴더 모두 자동 적용

### 결과

#### Before (문제 상황)
```
data/
├── train/ (이미지+마스크 섞여있음)
├── valid/ (이미지+마스크 섞여있음)  
└── test/ (이미지+마스크 섞여있음)
```

#### After (해결 완료)
```
data/
├── train/
│   ├── images/ (jpg 파일들)
│   └── masks/ (mask.png 파일들)
├── valid/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### 핵심 효과
- ✅ **파일 경로 오류 완전 해결**
- ✅ **데이터로더 안정적 동작**  
- ✅ **학습 과정 원활히 진행**

**결론**: 간단한 파일 분류 스크립트로 데이터셋 구조 문제를 해결하여 프로젝트를 원활하게 진행할 수 있었음!

---

## 📝 RunPod 환경에서 실행하기

[차선 인식 프로젝트 Colab 노트북](https://colab.research.google.com/drive/1mNNOflF0aAW2D52Q3m0ojeEksRXFK4pg#scrollTo=60006505-8d6d-4b0a-9adc-03e60aaffd15)  

위 링크를 통해 전체 구현 코드와 실행 결과를 확인할 수 있습니다.

### 📹 프로젝트 결과 영상
**차선 인식 모델 실행 결과 데모 영상 (30초)**

<p align="center">
  <img src="assets/lane_detection_demo.gif" alt="차선 인식 데모"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/🎬_차선_인식_결과-실시간_데모-4CAF50?style=for-the-badge&logo=videocam&logoColor=white" alt="차선 인식 결과"/>
  <br>
  <sub>📊 SegFormerForSemanticSegmentation 모델 추론 결과</sub>
</p>

---

### runpod Jupyter NoteBook에서 실행한 코드 상세 분석

### ✅ 실제 프로젝트 코드 분석 1

```
# ----------------- [ 1. 데이터셋 다운로드 ] -----------------
ROBOFLOW_API_KEY = "본인의 API-KEY 입력" # 본인의 API 키를 입력하세요.
WORKSPACE_ID = "jiyeonjin"
PROJECT_ID = "segmentation_-b4buk"
VERSION_NUMBER = 1

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
dataset = project.version(VERSION_NUMBER).download("png-mask-semantic")
base_data_path = dataset.location
clear_output()
print(f"✅ 데이터셋 다운로드 완료! 경로: {base_data_path}")
```
> Roboflow API로 데이터셋 다운로드
수동으로 zip 파일을 다운로드하고 업로드하는 번거로운 과정을 생략하고, **Roboflow API를 사용하면 코랩이나 런파드 환경으로 데이터셋을 직접, 그리고 훨씬 빠르게 다운로드**할 수 있습니다.

#### 1. API Key 발급받기
1.  Roboflow에 로그인한 후, 우측 상단의 프로필 아이콘을 클릭하여 **[Settings]**로 이동합니다.
2.  왼쪽 메뉴에서 **[Roboflow API]** 탭을 선택합니다.
3.  `Private API Key` 섹션에서 본인의 고유 API 키를 복사합니다. 이 키는 외부에 노출되지 않도록 주의해야 합니다.

    <img width="900" height="500" alt="image" src="https://github.com/user-attachments/assets/1aa12d1c-b59c-440c-a62c-2c66b8bc7bfe" />

#### 2. 노트북에서 API로 다운로드 실행
아래 코드를 노트북의 새로운 셀에 붙여넣고 실행하면, 데이터셋이 자동으로 다운로드 및 압축 해제됩니다.

```python
# 1. roboflow 라이브러리를 먼저 설치합니다.
!pip install -q roboflow

# 2. 복사한 코드를 붙여넣고 실행합니다.
from roboflow import Roboflow

# 본인의 API 키, 워크스페이스 ID, 프로젝트 ID, 버전 번호로 수정하세요.
rf = Roboflow(api_key="YOUR_PRIVATE_API_KEY")
project = rf.workspace("YOUR_WORKSPACE_ID").project("YOUR_PROJECT_ID")
dataset = project.version(YOUR_VERSION_NUMBER).download("png-mask-semantic")

# 다운로드된 경로를 확인합니다. (예: /content/lane-detection-1)
print(f"✅ 데이터셋 다운로드 완료! 경로: {dataset.location}")
```


### Segformer 차선 인식 프로젝트: 클라우드 환경 트러블슈팅 가이드
### **✅ 문제 1: 라이브러리 설치 실패 (`error: can't find Rust compiler`)**

Jupyter Notebook 환경에서 가장 먼저 마주칠 수 있는 문제입니다.

#### 오류 증상

`pip install` 명령어를 실행했을 때, 설치가 중단되며 아래와 유사한 붉은색 오류 메시지가 나타납니다.

```bash
error: subprocess-exited-with-error

× Building wheel for tokenizers (pyproject.toml) did not run successfully.
│ exit code: 1
╰─> [62 lines of output]
    ...
    running build_ext
    running build_rust
    error: can't find Rust compiler
    ...
ERROR: Failed to build installable wheels for some pyproject.toml based projects (tokenizers)
```

이 오류의 여파로, `from datasets import ...` 코드를 실행할 때 `ModuleNotFoundError: No module named 'datasets'` 라는 후속 오류가 발생합니다.

#### 🔍 원인 분석

이 문제의 핵심 원인은 `pip`가 **소스 코드를 직접 컴파일하여 라이브러리를 설치**하려고 시도하기 때문입니다.

- **의존성 문제**: `transformers` 라이브러리는 `tokenizers` 패키지에 의존하며, 이 패키지는 Rust 언어로 작성된 부분이 포함되어 있습니다.
- **버전 불일치**: 오래된 버전의 라이브러리(`transformers<4.21.0` 등)를 설치하도록 지정하면, 현재 사용 중인 최신 파이썬 환경과 호환되는 **미리 컴파일된 파일(wheel)**이 없을 수 있습니다.
- **컴파일 시도 및 실패**: 결국 `pip`는 소스 코드를 직접 컴파일하려고 시도하지만, 시스템에 **Rust 컴파일러**가 없으므로 빌드에 실패하고 오류를 발생시키는 것입니다.

#### 최종 해결책

가장 간단하고 올바른 해결책은 Rust 컴파일러를 설치하는 것이 아니라, **`pip`가 컴파일을 시도할 필요가 없도록 하는 것**입니다. 라이브러리 버전 제한을 제거하여 `pip`가 현재 환경에 가장 적합한 **미리 컴파일된 최신 버전을 자동으로 찾도록** 유도하면 됩니다.

**[수정 전]**
```python
!pip install -q "transformers<4.21.0" "datasets<2.4.0"
```

**[수정 후]**
```python
# 버전 제한을 모두 제거하여 최신 호환 버전을 설치합니다.
!pip install -q "transformers" "datasets" "accelerate" "opencv-python" "ipywidgets" "tqdm" "pillow" "roboflow"
```

---

### **✅ 문제 2: 훈련 후 영상 처리 실패 (`오류: 영상이 업로드되지 않았습니다.`)**

모델 훈련을 성공적으로 마친 후, 직접 촬영한 영상을 업로드하여 처리하려고 할 때 발생하는 가장 까다로운 문제입니다.

#### 오류 증상

`widgets.FileUpload` 위젯을 통해 영상을 업로드하고 처리 셀(6번 셀)을 실행하면, 영상 처리가 시작되지 않고 아래와 같은 오류 메시지만 출력됩니다.

```
❌ 오류: 영상이 업로드되지 않았습니다. 위 5번 셀에서 먼저 영상을 업로드해주세요.
```

결정적인 증거는 **파일 업로드 버튼의 숫자가 바뀌지 않는 현상**입니다. 브라우저에서 업로드가 100% 완료된 것처럼 보여도, 버튼이 `영상 업로드 (0)` 에서 `(1)`로 바뀌지 않았다면 이 문제가 발생한 것입니다.

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/42a0cded-089b-465e-9644-75c3e97b7eea" />


#### 🔍 원인 분석

이 문제는 코드의 논리 오류가 아니라, **웹 기반 Jupyter 환경과 브라우저 간의 통신 불안정성** 때문에 발생합니다.

- **파일 위젯의 한계**: `FileUpload` 위젯이 브라우저에서 서버(Jupyter 커널)로 파일 데이터를 전송하는 과정이 불안정하면, 파이썬 변수(`uploader.value`)에는 파일 정보가 제대로 등록되지 않을 수 있습니다.
- **결과**: 코드는 계속해서 비어있는 `uploader.value`를 확인하니, 파일이 없다고 판단할 수밖에 없습니다.

#### 최종 해결책 (단계별 가이드)

불안정한 업로드 위젯을 완전히 우회하고, **Jupyter 환경의 파일 탐색기를 통해 직접 파일을 업로드**하는 것이 가장 확실하고 실패 없는 방법입니다.

##### **1단계: 커널(Kernel) 재시작 - 환경 초기화**
변수 충돌이나 꼬임 현상을 방지하기 위해 커널을 재시작하여 환경을 깨끗하게 만듭니다. (**작성한 코드나 저장된 파일은 사라지지 않습니다**)

- **RunPod / JupyterLab**: 상단 메뉴 `[ Kernel ]` → `[ Restart Kernel... ]` 클릭
- **Google Colab**: 상단 메뉴 `[ 런타임 ]` → `[ 런타임 다시 시작 ]` 클릭
- **VS Code**: 노트북 오른쪽 상단의 **원형 화살표(🔄) 아이콘** 클릭

##### **2단계: 필수 셀만 재실행 (훈련 제외!)**
커널이 재시작되었으므로, 필요한 데이터와 라이브러리를 다시 메모리로 불러옵니다.
1. `[ 0. 환경 설정 ... ]` 셀 실행
2. `[ 1. 데이터셋 다운로드 ]` 셀 실행
3. `[ 2. 데이터셋 준비 ]` 셀 실행
4. `[ 3. 모델, 프로세서 ... ]` 셀 실행
5. **🚨 절대로 `[ 4. 모델 훈련 ]` 셀은 다시 실행하지 마세요! (가장 중요) -> 에포크(15분 정도 소요)가 재진행 됩니다.**

##### **3단계: 훈련된 모델 안전하게 불러오기**
커널 재시작으로 `trainer` 변수가 사라졌으므로, 디스크에 저장된 **훈련 완료된 모델**을 직접 불러오는 코드가 필요합니다. 아래 코드를 **새로운 셀**에 넣고 실행하세요.

```python
# ===================================================================
# [ ★ 새로운 셀 ★ ] - 저장된 모델 안전하게 불러오기
# ===================================================================
import glob

# 4번 셀(에포크 과정)을 다시 실행할 필요 없이, 훈련이 끝난 모델을 폴더에서 직접 불러옵니다.
# 가장 마지막에 저장된 체크포인트 폴더를 자동으로 찾아줍니다.
try:
    last_checkpoint = sorted(glob.glob("./segformer-b0-finetuned-lanes-final-v7/checkpoint-*/"))[-1]
    model = SegformerForSemanticSegmentation.from_pretrained(last_checkpoint)
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", do_reduce_labels=False)
    print(f"✅ 훈련이 완료된 모델 ({last_checkpoint})을 성공적으로 불러왔습니다.")
except IndexError:
    print("❌ 오류: 저장된 체크포인트 폴더를 찾을 수 없습니다. 훈련이 정상적으로 완료되었는지 확인해주세요.")
```

##### **4단계: 파일 직접 업로드 및 최종 처리**
이제 업로드 위젯을 사용하지 않습니다.

1.  **파일 직접 업로드**: 노트북 왼쪽의 **파일 탐색기** 창에서 **업로드 아이콘(↑)**을 눌러 처리할 영상을 직접 업로드합니다.
<img width="360" height="394" alt="image" src="https://github.com/user-attachments/assets/0648d6b6-1fbb-44df-b3b0-e6e2b717ab7a" />

3.  **최종 처리 코드 실행**: 기존의 5번, 6번 셀은 무시합니다. 아래 코드를 **새로운 셀**에 붙여넣고, `video_filename` 변수에 방금 업로드한 파일의 정확한 이름(왼쪽 상단의 workspace에서 경로 복사)을 입력한 뒤 실행하세요.

    ```python
    # ===================================================================
    # [ ★ 최종 해결용 셀 ★ ] - 직접 업로드된 파일 처리하기
    # ===================================================================
    from tqdm.notebook import tqdm
    from PIL import Image
    import numpy as np
    import torch, cv2, os

    # 1. 여기에 방금 직접 업로드한 영상 파일의 정확한 이름을 입력하세요.
    video_filename = "내영상.mp4"  # 예: "my_video.mp4"

    # 2. 아래 코드가 위 파일 이름으로 영상 처리를 시작합니다.
    print(f"✅ '{video_filename}' 파일을 직접 처리합니다. 잠시만 기다려주세요...")

    try:
        model.eval()
        input_filename = video_filename
        output_filename = "final_output_video.mp4"

        video_capture = cv2.VideoCapture(input_filename)
        if not video_capture.isOpened():
            print(f"❌ 오류: '{input_filename}' 파일을 열 수 없습니다. 파일 이름이 정확한지 확인하세요.")
        else:
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            print(f"\n✅ 영상 처리를 시작합니다... (총 {total_frames} 프레임)")

            for _ in tqdm(range(total_frames)):
                ret, frame = video_capture.read()
                if not ret: break
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                with torch.no_grad():
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    logits = model(**inputs).logits.cpu()
                
                mask = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).argmax(dim=1).numpy().astype(np.uint8)
                color_mask = np.zeros_like(frame)
                color_mask[mask == 1] = # 초록색 차선
                overlaid_frame = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)
                video_writer.write(overlaid_frame)
            
            video_capture.release()
            video_writer.release()
            
            print(f"\n✅ 영상 처리 완료! 결과가 '{output_filename}' 파일로 저장되었습니다.")

    except NameError:
        print("❌ 오류: 'model' 또는 'processor' 변수를 찾을 수 없습니다. 위 3단계 [★저장된 모델 불러오기★] 셀을 실행했는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")
    ```

> 최종 영상 출력 결과
<img width="900" height="800" alt="image" src="https://github.com/user-attachments/assets/06fedfac-a004-4197-952c-b9af38f94c33" />


## 프로젝트 마무리

> <p align="center">
  <img src="assets/lane_detection_demo2.gif" width="550" alt="Lane Detection Demo">
</p>

> 위의 오류 결과 영상 진단 :
> ### **학습 목표의 반전 현상**

> **모델이 '차선'이 아닌, '차선을 제외한 모든 배경(하늘, 나무 등)'을 학습했습니다.**

이것은 모델의 성능 문제가 아니라, 데이터의 **라벨 정의**와 모델의 **클래스 설정**이 서로 뒤바뀌어 발생한 문제입니다.

- **원인**: 모델에게는 **`1`번 픽셀이 차선**(`id2label = {1: "lane"}`)이라고 알려줬지만, 실제 훈련 데이터에서는 **배경이 `1`로, 차선이 `0`으로** 처리되었을 가능성이 매우 높습니다.
- **결과**: 모델은 배경(하늘, 나무)을 보면서 "이것이 차선이구나!"라고 잘못 학습했고, 그 결과 영상에서 배경 전체를 초록색으로 칠하게 된 것입니다.


### **해결을 위한 체크리스트**

#### **✅ 1단계: `id2label` 설정 뒤집기 (가장 빠른 해결책)**

훈련을 다시 할 필요 없이, **추론 코드에서 모델의 클래스 정의를 반대로 수정**하여 문제를 즉시 해결할 수 있습니다.

**[수정 전]**
```python
id2label = {0: "background", 1: "lane"}
```

**[수정 후]**
```python
# 0번 픽셀을 '차선'으로, 1번 픽셀을 '배경'으로 재정의
id2label = {0: "lane", 1: "background"} # ★ 이 부분이 핵심
```

#### **✅ 2단계: 데이터 전처리 함수(`final_transforms`) 결과 확인**

모델이 학습 직전에 받는 `labels` 텐서의 값을 직접 확인하여, 데이터가 의도대로 `0`과 `1`로 구성되어 있는지 최종 점검합니다.

---

이외 여러개의 결과 영상을 얻었습니다.
> <p align="center">
  <img src="assets/lane_detection_demo4.gif" width="550" alt="Lane Detection Demo 4">
</p>






