# 차선 인식 프로젝트 (SegFormerForSemanticSegmentation + 전이학습)
팀원 : 윤은식, 전은서, 박현욱, 유성일, 지연진

## 📌 프로젝트 개요
이 프로젝트는 **Hugging Face & NVIDIA 협업 SegFormerForSemanticSegmentation 모델**을 사용하여 **차선 인식(Lane Detection)** 을 수행합니다.  
`seg11xl.pt` 사전 학습 가중치를 활용해 **전이학습(Transfer Learning)** 으로 차선 픽셀 분류 모델을 학습합니다.

---

## 🛠 기술 스택
- **모델:** SegFormerForSemanticSegmentation (`seg11xl.pt` 기반)
- **데이터 라벨링:** Roboflow (Semantic Segmentation)
- **프로그래밍:** Python, PyTorch
- **환경:** Google Colab

---

## 📂 데이터 준비 (Roboflow)

### 1. 프로젝트 생성
1. Roboflow 접속 → `Create New Project`
2. **Project Type:** *Semantic Segmentation*
3. 프로젝트 이름: `lane-detection` (자유롭게 설정 가능)
4. 교수님께서 주신 영상 합쳐 업로드 (22분 가량)

---

### 2. 클래스 정의
> ⚠ **처음에는 단일 클래스 추천** → 데이터 수가 충분해지면 세부 클래스 추가 가능
> 우리팀의 경우 모든 차선을 'lane' 하나의 단일 클래스로만 간주

| 클래스명       | 설명                                    |
|----------------|----------------------------------------|
| `lane`         | 모든 차선 (색상/형태 관계없이)          |
| `lane_white`   | 흰색 차선 (선택사항)                    |
| `lane_yellow`  | 노란색 차선 (선택사항)                  |
| `lane_dashed`  | 점선 차선 (선택사항)                    |
| `lane_solid`   | 실선 차선 (선택사항)                    |

---

### 3. 라벨링 규칙
차선 픽셀을 정확하게 구분하는 것이 목표입니다.  
팀원분들은 다음 규칙을 따라 라벨링 해주세요.

#### ✅ 기본 규칙
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

#### ⚠ 주의사항
- 동일 장면에서 연속 프레임은 과도하게 포함하지 말 것 (데이터 중복 방지)
- 다양한 조건(맑음, 비, 야간, 역광, 그림자 포함)으로 데이터 확보
- 곡선 차선, 교차로 차선, 다차선 도로 등 다양한 형태 반영

---

### 4. 데이터 Export
라벨링 완료 후:
- **Export Format:** COCO Segmentation
- **Images:** JPG/PNG
- **Masks:** PNG (클래스별 색상 구분)
- **Train/Valid/Test Split:** 70% / 20% / 10% 추천

---

- # 📌 로보플로우 차선 인식 라벨링 & 마스크 생성 가이드

## 1. 폴리곤으로 라벨링
1. 라벨링할 이미지 클릭
2. 왼쪽 툴바에서 **Polygon Tool (펜 모양)** 선택
3. 차선 시작 지점부터 점을 찍어 영역을 만듦
   - 곡선은 점을 촘촘히 찍어 정확하게
4. 마지막 점에서 더블클릭 → 라벨 이름(`lane`) 선택
5. 저장 (`S` 키)

💡 **배경은 라벨링하지 않음** → 나중에 마스크 생성 시 자동으로 배경=0

---

## 2. 데이터셋 내보내기
1. 상단 메뉴 → **Export** 클릭
2. 포맷 선택: **Segmentation**
3. 형식: **PNG Masks**
4. 옵션 설정:
   - **One mask per image** ✅
   - **Class colors → Single-channel** ✅  
     (배경=0, 차선=1)
5. `Download Zip` 클릭

---

## 🧠 모델 학습 절차

1. **환경 준비**
```bash
pip install transformers datasets accelerate
