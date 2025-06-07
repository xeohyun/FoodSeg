# 🧠 FoodSeg  
**음식 이미지에서 음식 객체 분리 + LLM 기반 설명 및 칼로리 예측 시스템**

> 음식 사진 속 객체를 분할(Segmentation)하고, 각 객체에 대한 자연어 설명과 칼로리 예측을 수행합니다.

---

### 📸 Demo 
> (TODO: 추후 Gradio UI 동작 영상 추가)

---

## 💡 Motivation

이 프로젝트는 다음과 같은 실험적 시도와 시행착오를 기반으로 완성되었습니다:

- **Segment Anything Model (SAM)**, **Fast-SAM**, **DeepLabV3+** 등 다양한 segmentation 모델을 시도했지만, 음식 객체 간의 경계 구분이 미흡하거나 예측 라벨이 정확하지 않았습니다.
- 따라서, 음식에 특화된 dataset(FoodSeg103)에 fine-tuned 된 **Mask2Former** 모델을 한 번 더 fine-tuning하여 최종적으로 segmentation을 수행하였습니다.

---

### 🔪 Model Overview  

본 프로젝트에서는 **Mask2Former**를 기반으로 음식 분할(Semantic Segmentation)을 수행합니다.

- **기반 모델**: [Mask2Former](https://arxiv.org/abs/2112.01527)[논문](https://arxiv.org/pdf/2112.01527)
- **데이터셋**: [FoodSeg103](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103)
- **결과**: 음식 객체별 segmantation 마스크 및 lable 반환  
- **후처리**: LLM(Gemini)을 통해 자연어 설명 및 칼로리 예측

---

## 🔧 Model Architecture: Mask2Former

Mask2Former는 다음과 같은 구조로 구성되어 있습니다:

![Model Architecture](./mask2former_structure.png)

- **Backbone**: Swin Transformer 기반으로 hierarchical하게 feature를 추출합니다.
- **Pixel Decoder**: multi-scale feature를 통합하고 upsample하여 segmentation head로 전달합니다.
- **Transformer Decoder**: object query를 학습하며, 학습된 query에 대해 mask prediction을 수행합니다.
- **Segmentation Head**: binary mask 예측과 class 예측을 통해 최종 결과를 생성합니다.

---

## 🚀 주요 기능
- **Semantic Segmentation**: Mask2Former 기반 이미지 내 음식 분할
- **Gradio UI**: 웹 기반 이미지 업로드 및 결과 확인
- **Gemini API 연동**: 분할 결과 기반 자연어 설명 생성
- **모델 학습 기능**: Custom 학습 가능, yaml 기반 설정 파일 사용
- **모듈화된 구조**: 유지보수 및 기능 확장이 쉬운 구조로 설계

---
## 🔧 사용법

### 1. 프로젝트 구조
```
FoodSeg/
├── gradio_app/
│   ├── app.py                  # Gradio UI
│   └── model_inference.py      # mask 예측 함수
├── scripts/
│   └── train.py                # SegmentationTrainer 정의 (학습 로직)
│   └── run_training.py         # 모델 학습 스크립트
├── config.yaml                 # 학습 설정
├── foodseg_result/             # (학습된 모델 저장 경로)
└── README.md
```

### 2. 학습

```bash
python -m scripts.run_training --config configs/semantic/pipeline.yaml
```

> Checkpoint는 `foodseg_result/` 하위에 `.pth` 파일로 저장됨

### 3. 실행

```bash
python gradio_app/app.py
```

> Gradio UI를 통해 이미지 입력 시 마스크 결과 + 자연어 설명 + 칼로리 예측까지 확인 가능

---
## 📂 데이터셋 구성

- **이름**: FoodSeg103 (HuggingFace에서 사용 가능)

- **Train 데이터 수**: 약 6,000장

- **Validation 데이터 수**: 약 1,600장

- **총 클래스 수**: 104개 (음식 종류별 고유 라벨 포함)

### 🏷️ 라벨 (id2label 포맷)

FoodSeg103 데이터셋은 다음과 같은 형식의 라벨을 포함합니다:

- 0: background

- 1: apple

- 2: banana

- 3: fried rice

...

- 103: yogurt


각 이미지의 픽셀은 이 라벨 ID에 해당하는 값을 가지며, `.json` 형식으로 제공되는 라벨 매핑 파일(`id2label.json`)을 통해 사람이 읽을 수 있는 라벨명으로 변환됩니다.

> 예: `id2label[3]` → `"fried rice"`

---

## 🧪 코드 구성 및 설명

### 1. `gradio_app/model_inference.py`
- **목적**: 사용자 이미지 업로드 시 segmentation 예측 + Gemini 설명 생성
- **구성 기능**:
  - `load_model_and_processor()`: 가장 최신 checkpoint 불러오기
  - `predict_masks()`: 이미지 전처리, 모델 추론, 마스크 시각화 수행
  - `generate_caption_from_labels_with_calories()`: 인식된 라벨을 기반으로 Gemini API 호출하여 텍스트 생성

### 2. `scripts/run_training.py`
- **목적**: 학습 설정을 담은 `config.yaml`을 불러와 FoodSeg103 데이터셋으로 Mask2Former 모델 학습 수행
- **주요 기능**:
  - `load_config()`: yaml 로부터 설정 로딩
  - HuggingFace Hub에서 `id2label` 및 `train/val dataset` 다운로드
  - `SegmentationTrainer` 인스턴스 생성 및 학습 시작

### 3. `scripts/train.py`
- **목적**: 학습 로직이 정의된 클래스 `SegmentationTrainer` 구현
- **주요 기능**:
  | 기능 | 설명 |
  |------|------|
  | 데이터 로딩 및 전처리 | `Albumentations` 이용해 Resize, Normalize, Flip 등 적용 |
  | 모델 초기화 | 사전학습된 `mask2former-swin-small-ade-semantic` 불러옴 |
  | 학습 루프 | Epoch별 학습, Loss 기록, Validation 포함 |
  | 평가 지표 | mean IoU(`evaluate` 라이브러리) 사용 |
  | 모델 저장 | `save_pretrained()` + `.pth` 로 저장 |
  | 시각화 | `tensorboard` 연동 로그 저장 |

---

## 📈 Training Performance Summary

| Epoch | Avg Training Loss | Validation Loss | Mean IoU |
|-------|-------------------|------------------|----------|
| 1     | 45.7836           | 36.5247          | 0.1157   |
| 2     | 31.0042           | 29.8534          | 0.1727   |
| 3     | 24.7041           | 26.1080          | 0.2254   |

- **Model Checkpoints Saved:** After each epoch
- **Best Epoch (so far):** Epoch 2

> TensorBoard를 통해 첫 번째 epoch에서의 손실 값을 확인한 결과,  
> 학습 손실은 약 **45.85**, 검증 손실은 약 **35.67** 수준이었음.  
> 아직 학습 초기 단계이며, 더 많은 epoch가 필요함.

---

## ⚠️ 한계점

- 모델 성능이 아직 낮음
- 초기에는 다양한 모델 (SAM, Segment Anything 등) 시도했으나,  
  음식 세그먼트에 적합하지 않아서 최종적으로 Mask2Former 참조
- 정확한 칼로리 계산을 위해서는 실제 물리 크기나 무게 정보가 필요하지만,  
  현재는 단순 **라벨 기반 대략 예측** 수준
- Segmentation의 pixel 값과 depthmap을 함께 사용해 실제 음식 크기를 추정하고 칼로리를 예측하려 했으나,  
  구현 단계에서는 정확한 depth 정보와의 정합이 어려웠음
- **라벨 클래스 수가 제한적임 (총 104개 클래스)**  
  → 실제 존재하는 수천 가지 음식 종류를 모두 커버하지 못함  
  → 예: 김치찌개, 떡볶이, 잡채 등의 **일상적 한국 음식은 미포함**  
  → 이로 인해 **분류 불가능하거나 부정확한 예측**이 발생할 수 있음
---

## 🚧 향후 계획

- GPU 환경에서 장시간 학습 → 성능 향상
- Fine-tuning 데이터셋 증가
- 실제 칼로리 계산을 위한 depth estimation 또는 reference scaling 도입

---
