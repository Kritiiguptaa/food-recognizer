# app.py — Predict BOTH cuisines

from pathlib import Path
import io, glob, difflib, random
import torch, torch.nn as nn
from PIL import Image
import pandas as pd
import streamlit as st
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

st.set_page_config(page_title="Indian Food Recognizer", page_icon="🍽️", layout="wide")
st.title("🍽️ Indian Food Recognizer")
st.caption("Upload an image → Get predictions from both Punjabi & South Indian models.")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CUISINE_ROOT = ROOT / "cuisines"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Expected folder structure:
# cuisines/
#   Punjabi/
#     model/best_efficientnet_b0.pth
#     model/classes.txt
#     recipes.xlsx
#     data/Punjabi_Split_Data/...
#   SouthIndian/
#     model/best_efficientnet_b0.pth
#     model/classes.txt
#     recipes.xlsx
#     data/SouthIndian_Split_Data/...

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower().strip() if ch.isalnum() or ch.isspace())

@st.cache_resource
def load_classes(path):
    return [line.strip() for line in open(path, "r", encoding="utf-8")]

@st.cache_resource
def load_model(path, num_classes: int):
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    return m

@st.cache_data
def load_excel(path):
    df = pd.read_excel(path)
    df["Recipe Name"] = df["Recipe Name"].astype(str)
    return df

TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict(img, model, classes, tta=3):
    x = TFM(img.convert("RGB")).unsqueeze(0).to(device)

    logits = 0
    for i in range(tta):
        logits += model(torch.flip(x, dims=[3]) if i % 2 else x)

    probs = (logits / tta).softmax(1).squeeze(0)
    p, idx = probs.topk(5)

    return [(classes[i], float(pp)) for pp, i in zip(p.tolist(), idx.tolist())]

def lookup_recipe(name, df):
    key = _norm(name)
    for recipe in df["Recipe Name"]:
        if _norm(recipe) == key:
            return df[df["Recipe Name"] == recipe].iloc[0].to_dict()
    return None

# ------------------------------------------------------------------
# Load BOTH cuisines
# ------------------------------------------------------------------
def load_cuisine(cuisine_name):
    C = CUISINE_ROOT / cuisine_name
    model = load_model(C / "models" / "best_efficientnet_b0.pth",
                       len(load_classes(C / "models" / "classes.txt")))
    
    return {
        "name": cuisine_name,
        "model": model,
        "classes": load_classes(C / "models" / "classes.txt"),
        "recipes": load_excel(C / "recipes.xlsx"),
        "data_root": C / "data"
    }

ALL_CUISINES = {
    "Punjabi": load_cuisine("Punjabi Cuisines"),
    "South Indian": load_cuisine("South Indian Cuisines")
}

# ------------------------------------------------------------------
# Main UI
# ------------------------------------------------------------------
colL, colR = st.columns([1, 1])
uploaded = colL.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
cam = colR.camera_input("Take Photo")

tta = st.slider("Prediction averaging (TTA)", 1, 7, 3)

# ------------------------------------------------------------------
# Run predictions
# ------------------------------------------------------------------
if uploaded or cam:
    file = cam if cam else uploaded
    img = Image.open(io.BytesIO(file.getvalue()))

    # Display smaller image
    st.image(img, caption="Input Image", use_column_width=False, width=350)

    results = {}

    # Predict for each cuisine
    for cname, C in ALL_CUISINES.items():
        top5 = predict(img, C["model"], C["classes"], tta=tta)
        results[cname] = top5

    # Display results side-by-side
    # st.markdown("## 🔍 Predictions from both cuisines")

        # ------------------------------------------------------------------
    # Decide cuisine automatically
    # ------------------------------------------------------------------

    south_top5 = results["South Indian"]
    punjabi_top5 = results["Punjabi"]

    best_south = south_top5[0][1]   # confidence
    best_punjabi = punjabi_top5[0][1]

    if best_south >= best_punjabi:
        final_cuisine = "South Indian"
        final_preds = south_top5
    else:
        final_cuisine = "Punjabi"
        final_preds = punjabi_top5

    # ------------------------------------------------------------------
    # Show only the detected cuisine
    # ------------------------------------------------------------------

    st.markdown(f"## 🍽 Predicted Cuisine: **{final_cuisine}**")

    st.table(pd.DataFrame(final_preds, columns=["Dish", "Confidence"]))

    top1_name = final_preds[0][0]

    # ------------------------------------------------------------------
    # Recipe Details
    # ------------------------------------------------------------------

    st.markdown("## 📘 Recipe Details")

    Cbest = ALL_CUISINES[final_cuisine]
    recipe_df = Cbest["recipes"]

    info = lookup_recipe(top1_name, recipe_df)

    st.markdown(f"### Best match: **{top1_name}**")

    if info:
        for k, v in info.items():
            st.write(f"**{k}:** {v if v not in ['', None] else '-'}")
    else:
        st.info("No recipe found in the dataset.")
