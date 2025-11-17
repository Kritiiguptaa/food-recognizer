# app.py  — improved UX
from pathlib import Path
import io, glob, difflib, random
import torch, torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import pandas as pd
import streamlit as st

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent
MODEL_PATH   = ROOT / "models" / "best_efficientnet_b0.pth"
CLASSES_PATH = ROOT / "models" / "classes.txt"
EXCEL_PATH   = ROOT / "Punjabi food.xlsx"
DATA_SPLIT   = ROOT / "data" / "Punjabi_Split_Data"  # to fetch sample images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- helpers ----------
def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower().strip() if ch.isalnum() or ch.isspace())

# @st.cache(allow_output_mutation=True)
@st.cache_resource
def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return [c.strip() for c in f if c.strip()]

# @st.cache(allow_output_mutation=True)
@st.cache_resource
def load_model(num_classes: int):
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.to(device).eval()
    return m

# @st.cache(allow_output_mutation=True)
@st.cache_data
def load_excel():
    df = pd.read_excel(EXCEL_PATH)
    df["Recipe Name"] = df["Recipe Name"].astype(str)
    return df

TFM = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

@torch.no_grad()
def predict_pil(img: Image.Image, model, classes, tta:int=1):
    x = TFM(img.convert("RGB")).unsqueeze(0).to(device)
    logits = 0
    for i in range(max(1,tta)):
        x_aug = torch.flip(x, dims=[3]) if (i % 2 == 1) else x
        logits = logits + model(x_aug)
    probs = (logits / float(max(1,tta))).softmax(1).squeeze(0)
    p, idx = probs.topk(5)
    top5 = [(classes[i], float(pp)) for pp,i in zip(p.tolist(), idx.tolist())]
    return top5

def lookup_excel(pred_name: str, df: pd.DataFrame):
    key = _norm(pred_name)
    # exact normalized match first
    for recipe in df["Recipe Name"]:
        if _norm(recipe) == key:
            row = df[df["Recipe Name"] == recipe].iloc[0]
            return {col: (None if pd.isna(row[col]) else row[col]) for col in df.columns}
    # fuzzy fallback
    candidates = [ _norm(r) for r in df["Recipe Name"].tolist() ]
    m = difflib.get_close_matches(key, candidates, n=1, cutoff=0.6)
    if m:
        idx = candidates.index(m[0])
        row = df.iloc[idx]
        return {col: (None if pd.isna(row[col]) else row[col]) for col in df.columns}
    return None

def sample_image_for_class(class_name: str):
    """Find a representative image from dataset (test→val→train)."""
    safe = class_name  # folder names equal to class names in your split
    patterns = [
        DATA_SPLIT / "test"  / safe / "*.*",
        DATA_SPLIT / "val"   / safe / "*.*",
        DATA_SPLIT / "train" / safe / "*.*",
    ]
    for pat in patterns:
        files = [f for f in glob.glob(str(pat)) if f.lower().endswith((".jpg",".jpeg",".png",".webp"))]
        if files:
            return random.choice(files)
    return None

def pretty_info_block(info: dict):
    # Choose preferred columns if present, else show all
    preferred = ["Recipe Name","Category","Ingredients","Instructions","Serving Size","Calories","Preparation Time","Cooking Time"]
    show = [c for c in preferred if c in info] or list(info.keys())
    for col in show:
        val = info[col]
        if isinstance(val, str) and len(val) > 1200:
            val = val[:1200] + " …"
        st.markdown(f"**{col}**")
        st.write(val if val not in [None, ""] else "-")

# ---------- UI ----------
st.set_page_config(page_title="Punjabi Food Recognizer", page_icon="🍛", layout="wide")

st.title("🍛 Punjabi Food Recognizer")
st.caption("Upload a photo or use your camera. Get the dish and its recipe details.")

colL, colR = st.columns(2)
with colL:
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
with colR:
    cam_img = st.camera_input("Or take a photo")

tta = st.slider("TTA (averaged predictions)", 1, 7, 5, help="Higher can be a bit more accurate, slightly slower.")

if uploaded or cam_img:
    # prefer the latest provided image
    file = cam_img if cam_img else uploaded
    img = Image.open(io.BytesIO(file.getvalue())).convert("RGB")

    classes = load_classes()
    model = load_model(len(classes))
    df = load_excel()

    # --- prediction ---
    top5 = predict_pil(img, model, classes, tta=tta)
    top1_name, top1_conf = top5[0]

    # LEFT: prediction + selector + top-5 table
    left, right = st.columns([1,1.2])
    with left:
        st.subheader("Prediction")
        st.markdown(f"<h2 style='margin-top:-10px'>{top1_name}</h2>", unsafe_allow_html=True)
        st.metric(label="Confidence", value=f"{top1_conf:.3f}")

        # clickable choice among Top-5 (default Top-1)
        options = [name for name, _ in top5]
        selected = st.selectbox("View recipe details for:", options, index=0)

        st.write("Top-5 probabilities")
        st.table(pd.DataFrame(top5, columns=["Dish","Confidence"]))

    # RIGHT: dataset image first, then info for selected dish
    with right:
        # dataset sample (fallback to uploaded image)
        sample_path = sample_image_for_class(selected)
        if sample_path and Path(sample_path).exists():
            st.image(sample_path, caption=f"Sample from dataset: {selected}", use_column_width=True)
        else:
            st.image(img, caption="Input", use_column_width=True)

        st.subheader("Recipe Info")
        info = lookup_excel(selected, df)
        if info:
            pretty_info_block(info)
        else:
            st.info("No matching entry found in the spreadsheet for this dish.")
else:
    st.info("Upload an image or use the camera to get started.")
