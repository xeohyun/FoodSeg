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
- 결국 음식에 특화된 dataset(FoodSeg103)에 pretrained된 **Mask2Former** 모델을 fine-tuning하여 최종적으로 적용하였습니다.
- 로컬 Mac 환경에서 **CPU로만 학습을 진행**해야 했기에 시간이 오래 걸렸으며, 아직까지 성능이 높은 수준은 아닙니다.

---

### 🔪 Model Overview  

본 프로젝트에서는 **Mask2Former**를 기반으로 음식 분할(Semantic Segmentation)을 수행합니다.

- **기반 모델**: [Mask2Former](https://arxiv.org/abs/2112.01527)
- **데이터셋**: [FoodSeg103](https://github.com/lsy17096535/FoodSeg103)
- **결과**: 음식 객체별 마스크와 라벨 반환  
- **후처리**: LLM(Gemini)을 통해 자연어 설명 및 칼로리 예측

---

## 🔧 Model Architecture: Mask2Former

Mask2Former는 다음과 같은 구조로 구성되어 있습니다:

![Model Architecture](./mask2former_structure.png)

- **Backbone**: Swin Transformer 기반으로 hierarchical하게 feature를 추출합니다.
- **Pixel Decoder**: multi-scale feature를 통합하고 upsample하여 segmentation head로 전달합니다.
- **Transformer Decoder**: object query를 학습하며, 각 query에 대해 mask prediction을 수행합니다.
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
├── foodseg_result/                # (학습된 모델 저장 경로)
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

## ⚠️ 한계점

- 모델 성능이 아직 낮음 (**현재 1 epoch 만 학습됨**)
- 실험 환경이 GPU가 아니는 **CPU 기반**, 학습 속도 매우 느린
- 초기에는 다양한 모델 (SAM, Segment Anything 등) 시도했으나,  
  음식 세그먼트에 적합하지 않아서 최종적으로 Mask2Former 참조
- 정확한 칼로리 계산을 위해서는 실제 물리 크기나 무게 정보가 필요하지만,  
  현재는 단순 **라벨 기반 대량 예측** 수준

---

## 🚧 향후 계획

- GPU 환경에서 장시간 학습 → 성능 향상
- Fine-tuning 데이터셋 증가
- 실제 칼로리 계산을 위한 depth estimation 또는 reference scaling 도입

---
