import os
import json
import random
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from huggingface_hub import hf_hub_download
import albumentations as A


def color_palette(num_classes=150, seed=85):
    """     
    Generates a consistent color palette for a given number of classes by setting a random seed.

    Args:
        num_classes (int): Number of classes/colors to generate.
        seed (int): Seed for the random number generator.

    Returns:
        list: A list of RGB values.
    """
    random.seed(seed)
    palette = []
    for _ in range(num_classes):
        color = [random.randint(0, 255) for _ in range(3)]
        palette.append(color)
    return palette


def load_model_and_processor(device):
    """
    Loads the Mask2Former model and processor from a valid directory only.
    """
    directory_path = "/Users/xeohyun/DEV/CV/FoodSeg_mask2former/foodseg_result"
    all_subdirs = [
        d for d in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, d)) and
        (
            os.path.exists(os.path.join(directory_path, d, "pytorch_model.bin")) or
            os.path.exists(os.path.join(directory_path, d, "model.safetensors"))
        )
    ]

    if not all_subdirs:
        raise FileNotFoundError("❌ .bin 또는 .safetensors가 있는 폴더를 foodseg_result 안에서 찾을 수 없습니다.")

    latest_checkpoint = sorted(all_subdirs)[-1]
    saved_model_path = os.path.join(directory_path, latest_checkpoint)
    print(f"📦 모델 로딩 경로: {saved_model_path}")

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        saved_model_path, use_safetensors=True
    ).to(device)

    processor = Mask2FormerImageProcessor(
        ignore_index=0,
        reduce_labels=False,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )

    return model, processor



def visualize_panoptic_segmentation(
    original_image_np, segmentation_mask, segments_info, category_names
):
    """
    Visualizes the segmentation mask overlaid on the original image with category labels.

    Args:
        original_image_np (np.ndarray): The original image in NumPy array format.
        segmentation_mask (np.ndarray): The segmentation mask.
        segments_info (list): Information about the segments.
        category_names (list): List of category names corresponding to segment IDs.

    Returns:
        PIL.Image.Image: The overlayed image with segmentation mask and labels.
    """
    # Create a blank image for the segmentation mask
    segmentation_image = np.zeros_like(original_image_np)

    num_classes = len(category_names)
    palette = color_palette(num_classes)

    # Apply colors to the segmentation mask
    for segment in segments_info:
        if segment["label_id"] == 0:
            continue
        color = palette[segment["label_id"]]
        mask = segmentation_mask == segment["id"]
        segmentation_image[mask] = color

    # Overlay the segmentation mask on the original image
    alpha = 0.5  # Transparency for the overlay
    overlay_image = cv2.addWeighted(
        original_image_np, 1 - alpha, segmentation_image, alpha, 0
    )

    # Convert to PIL image for text drawing
    overlay_image_pil = Image.fromarray(overlay_image)
    draw = ImageDraw.Draw(overlay_image_pil)

    # Set up font size
    base_font_size = max(
        20, int(min(original_image_np.shape[0], original_image_np.shape[1]) * 0.03)
    )

    # Set up font path
    from PIL import ImageFont

    # Optional: Load custom font
    try:
        font_path = "/Library/Fonts/Arial.ttf"
        font = ImageFont.truetype(font_path, base_font_size)
    except IOError:
        font = ImageFont.load_default()
        print("⚠️ Custom font not found. Using default font instead.")


    # Draw category labels on the image
    for segment in segments_info:
        label_id = segment.get("label_id")
        if label_id is not None and 0 <= label_id < len(category_names):
            category = category_names[label_id]
            mask = (segmentation_mask == segment["id"]).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            if num_labels > 1:
                largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                centroid_x = int(centroids[largest_component][0])
                centroid_y = int(centroids[largest_component][1])

                # Ensure text is within image bounds
                text_position = (
                    max(0, min(centroid_x, original_image_np.shape[1] - 1)),
                    max(0, min(centroid_y, original_image_np.shape[0] - 1)),
                )
                draw.text(text_position, category, fill=(0, 0, 0), font=font)

    return overlay_image_pil

from PIL import ImageOps 

def predict_masks(input_image_path):
    """
    Predicts and visualizes segmentation masks for a given image.

    Args:
        input_image_path (str): Path to the input image.

    Returns:
        PIL.Image.Image: The image with overlaid segmentation mask and labels.
    """
    # Determine device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model_and_processor(device)

    # Load category labels
    repo_id = "EduardoPacheco/FoodSeg103"
    filename = "id2label.json"
    id2label = json.load(
        open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r")
    )
    id2label = {int(k): v for k, v in id2label.items()}

    # Load and preprocess image
    image_PIL = Image.open(input_image_path)
    image_PIL = ImageOps.exif_transpose(image_PIL)
    original_image_np = np.array(image_PIL)

    transform = A.Compose(
        [
            A.Resize(width=512, height=512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transformed = transform(image=original_image_np)
    image = transformed["image"]

    # Convert image to C, H, W format
    image = image.transpose(2, 0, 1)

    # Process the image and get predictions
    inputs = processor([image], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image_PIL.size[::-1]]
    )[0]
    segmentation_mask = result["segmentation"].cpu().numpy()

    segments_info = result["segments_info"]
    output_result = visualize_panoptic_segmentation(
        original_image_np, segmentation_mask, segments_info, id2label
    )
     # 📌 예측된 label 목록 추출
    predicted_labels = []
    for segment in segments_info:
        label_id = segment.get("label_id")
        if label_id and label_id in id2label:
            label = id2label[label_id]
            if label not in predicted_labels:
                predicted_labels.append(label)

    # 📌 OpenAI로 설명 생성
    description = generate_caption_from_labels_with_calories(predicted_labels, input_image_path)

    return output_result, description

import os
from dotenv import load_dotenv
import google.generativeai as genai

# 🔑 Gemini API 키 설정 (환경변수에서 가져오기)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def generate_caption_from_labels_with_calories(labels,input_image_path):
    if not labels:
        return "음식이 감지되지 않았습니다."

    prompt = f"""
아래는 이미지에서 감지된 재료 목록입니다:  
{', '.join(labels)}

1. 이 재료들을 기반으로 예상되는 음식이 있다면 하나만 구체적으로 추정해 주세요.  
2. 해당 음식이 일반적으로 가지는 평균 칼로리를 추정해 주세요.  
→ 예: **햄버거 - 약 500kcal**

3. 이후에는 위에 감지된 **개별 재료들 각각에 대해** 다음 정보를 제공해 주세요:  
    - 구성 설명  
    - 1인분 혹은 일반적인 양 기준의 칼로리  
    - 간단한 참고 정보  
형식은 아래 예시처럼 통일해 주세요.

---

예시 출력:

**예상 음식:** 햄버거 — 약 500kcal

**재료별 정보:**  
- **빵 (Bread)** (주요 구성: 밀가루, 물 등) — 약 200kcal  
- **소고기 패티 (Beef Patty)** (100g 기준) — 약 250kcal  
- **치즈 (Cheese)** — 약 80kcal  

※ 위 정보는 일반적인 기준이며 실제 조리 방식, 양, 재료에 따라 달라질 수 있습니다.
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        images = [Image.open(input_image_path)]
        response = model.generate_content(
            [prompt] + images,
            generation_config={"temperature": 0.7}
            )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini API 호출 실패: {str(e)}"
