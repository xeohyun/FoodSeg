import os
import torch
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

def convert_and_save():
    ckpt_path = "/Users/xeohyun/DEV/CV/FoodSeg_mask2former/foodseg_result/epoch_1/model_epoch.pth"
    save_path = "/Users/xeohyun/DEV/CV/FoodSeg_mask2former/foodseg_result/epoch_1"
    os.makedirs(save_path, exist_ok=True)

    # 정확한 class 수로 config 생성
    config = Mask2FormerConfig.from_pretrained(
        "facebook/mask2former-swin-large-coco-panoptic"
    )
    config.num_labels = 105  # ← 여기가 핵심!

    model = Mask2FormerForUniversalSegmentation(config)

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # 만약 "module." prefix가 있다면 제거
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("⚠️ 누락된 키:", missing)
    print("⚠️ 예상치 못한 키:", unexpected)

    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    model.config.to_json_file(os.path.join(save_path, "config.json"))
    print("✅ 변환 완료!")

if __name__ == "__main__":
    convert_and_save()
