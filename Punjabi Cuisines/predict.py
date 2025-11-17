import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import pandas as pd
import difflib
import argparse
import json
from pathlib import Path

# ================= PATHS =================
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "best_efficientnet_b0.pth"
CLASSES_PATH = BASE / "models" / "classes.txt"
EXCEL_PATH = BASE / "Punjabi food.xlsx"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= HELPERS ================
def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return [c.strip() for c in f if c.strip()]


def create_model(num_classes):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def normalize_name(s):
    return "".join(ch for ch in s.lower().strip() if ch.isalnum() or ch.isspace())


# Transform (same as test)
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])


@torch.no_grad()
def predict_tensor(model, x, tta=1):
    model.eval()
    logits = 0
    for i in range(tta):
        aug = x.flip(3) if (i % 2 == 1) else x
        logits += model(aug)
    logits /= tta
    return logits.softmax(1).squeeze(0)


def predict_image(image_path, tta=1):
    classes = load_classes()
    model = create_model(len(classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    probs = predict_tensor(model, x, tta=tta)
    top5_p, top5_idx = probs.topk(5)

    top1 = (classes[top5_idx[0]], float(top5_p[0]))
    top5 = [(classes[idx], float(p)) for p, idx in zip(top5_p.tolist(), top5_idx.tolist())]

    # Excel lookup
    df = pd.read_excel(EXCEL_PATH)
    df["Recipe Name"] = df["Recipe Name"].astype(str)
    key = normalize_name(top1[0])

    match = None
    for recipe in df["Recipe Name"]:
        if normalize_name(recipe) == key:
            match = recipe
            break

    if not match:
        possibilities = [normalize_name(r) for r in df["Recipe Name"]]
        found = difflib.get_close_matches(key, possibilities, n=1, cutoff=0.6)
        if found:
            match = df["Recipe Name"].iloc[possibilities.index(found[0])]

    info = None
    if match:
        row = df[df["Recipe Name"] == match].iloc[0]
        info = {col: None if pd.isna(val) else val for col, val in row.items()}

    return {"top1": top1, "top5": top5, "excel": info}


# ================ CLI ================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Input food image")
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    result = predict_image(args.image, args.tta)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("\n✅ Prediction")
    print(f"Top-1: {result['top1'][0]} (conf: {result['top1'][1]:.3f})")
    print("Top-5:")
    for name, conf in result["top5"]:
        print(f" - {name}: {conf:.3f}")

    if result["excel"]:
        print("\n📖 Excel Info:")
        for k, v in result["excel"].items():
            print(f"  {k}: {v}")
    else:
        print("\n⚠ Found no matching entry in Excel")


if __name__ == "__main__":
    main()
