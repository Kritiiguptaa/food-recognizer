import os, requests, pandas as pd, time
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
assert SERPAPI_KEY, "Missing SERPAPI_KEY"

base_dir = "SouthIndian_Recipe_Images"
os.makedirs(base_dir, exist_ok=True)

def safe_name(name: str) -> str:
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, '')
    return name.strip()

def make_queries(recipe_name: str):
    base = recipe_name
    return [
        f"{base} South Indian dish",
        f"{base} South Indian recipe",
        f"{base} restaurant plate",
        f"{base} home cooked",
    ]

def fetch_google_images(query: str, num_needed: int, max_pages: int = 5):
    urls = []
    seen = set()
    per_page = min(100, num_needed)
    headers = {"User-Agent": "Mozilla/5.0"}

    for page in range(max_pages):
        if len(urls) >= num_needed:
            break
        params = {
            "engine": "google",
            "q": query,
            "tbm": "isch",
            "ijn": page,
            "num": per_page,
            "api_key": SERPAPI_KEY
        }
        try:
            r = requests.get("https://serpapi.com/search.json",
                             params=params, timeout=30, headers=headers)
            data = r.json()
            for img in data.get("images_results", []):
                u = img.get("original")
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
                if len(urls) >= num_needed:
                    break
        except Exception as e:
            print(f"⚠️ SerpApi page {page} failed for '{query}': {e}")
            continue
        time.sleep(0.3)
    return urls[:num_needed]

def download_image(url, path):
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if min(img.size) < 120:
            return False
        img.save(path, "JPEG", quality=90)
        return True
    except Exception:
        return False

# Load South Indian recipes
# df = pd.read_excel("cuisines/South Indian Cuisines/recipes.xlsx")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # go 2 levels up to ml_model/
EXCEL_PATH = ROOT / "cuisines" / "South Indian Cuisines" / "recipes.xlsx"

df = pd.read_excel(EXCEL_PATH)
print("Loaded:", EXCEL_PATH)

TARGET_PER_CLASS = 200

for recipe in df["Recipe Name"].dropna().unique():
    folder = os.path.join(base_dir, safe_name(recipe))
    os.makedirs(folder, exist_ok=True)

    existing = len([f for f in os.listdir(folder)
                    if f.lower().endswith((".jpg",".jpeg",".png"))])
    if existing >= TARGET_PER_CLASS:
        print(f"✅ Already have {existing} images for {recipe}")
        continue

    need = TARGET_PER_CLASS - existing
    print(f"📥 Need {need} more images for {recipe}")

    queries = make_queries(recipe)
    all_urls, q_idx = [], 0
    while len(all_urls) < need and q_idx < len(queries):
        left = need - len(all_urls)
        urls = fetch_google_images(queries[q_idx], num_needed=left, max_pages=5)
        all_urls.extend(urls)
        q_idx += 1

    print(f"🔍 Collected {len(all_urls)} URLs for {recipe}")

    saved = 0
    start_idx = existing + 1
    for url in all_urls:
        filename = f"{start_idx + saved:04d}.jpg"
        out_path = os.path.join(folder, filename)
        if download_image(url, out_path):
            saved += 1
        time.sleep(0.2)

    print(f"✅ Saved {saved} images for {recipe} (total {existing + saved})")

print("\n🎉 All South Indian Recipes processed successfully!")
