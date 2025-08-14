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

## 5. 다운로드 한 데이터셋을 가지고 코드 실행하기

### 📝 코랩 환경에서 실행하기
[차선 인식 프로젝트 코랩 노트북](https://colab.research.google.com/github/jiyeonjin/0624_new/blob/main/0813_%ED%8C%80%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8.ipynb)

위 링크를 통해 전체 구현 코드와 실행 결과를 확인할 수 있습니다.


### 📹 프로젝트 결과 영상
**차선 인식 모델 실행 결과 데모 영상 (2분)**

![차선 인식 데모]아래 영상에서 실제 차선 인식 모델의 추론 결과를 확인할 수 있습니다:

아래 영상에서 실제 차선 인식 모델의 추론 결과를 확인할 수 있습니다:

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


#### 실제 프로젝트 코드 분석

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

---

### 📝 runpod 환경에서 실행하기
