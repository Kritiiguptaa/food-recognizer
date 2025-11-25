import os, requests, pandas as pd, time
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
assert SERPAPI_KEY, "Missing SERPAPI_KEY"

base_dir = "Punjabi_Recipe_Images"
os.makedirs(base_dir, exist_ok=True)

def safe_name(name: str) -> str:
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, '')
    return name.strip()

def make_queries(recipe_name: str):
    base = recipe_name
    return [
        f"{base} Punjabi dish",
        f"{base} Punjabi recipe",
        f"{base} restaurant plate",
        f"{base} home cooked",
    ]

def fetch_google_images(query: str, num_needed: int, max_pages: int = 5):
    """Get >100 by paging: ijn=0,1,2... Each page returns up to ~100."""
    urls = []
    seen = set()
    per_page = min(100, num_needed)  # Google caps around 100/page
    headers = {"User-Agent": "Mozilla/5.0"}

    for page in range(max_pages):
        if len(urls) >= num_needed:
            break
        params = {
            "engine": "google",
            "q": query,
            "tbm": "isch",
            "ijn": page,           # <-- pagination
            "num": per_page,
            "api_key": SERPAPI_KEY
        }
        try:
            r = requests.get("https://serpapi.com/search.json", params=params, timeout=30, headers=headers)
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
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        # Skip tiny thumbnails
        if min(img.size) < 120:
            return False
        img.save(path, "JPEG", quality=90)
        return True
    except Exception:
        return False

# Load list
df = pd.read_excel("Punjabi food.xlsx")

TARGET_PER_CLASS = 200

for recipe in df["Recipe Name"].dropna().unique():
    folder = os.path.join(base_dir, safe_name(recipe))
    os.makedirs(folder, exist_ok=True)

    existing = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))])
    if existing >= TARGET_PER_CLASS:
        print(f"✅ Already have {existing} images for {recipe}")
        continue

    need = TARGET_PER_CLASS - existing
    print(f"📥 Need {need} more images for {recipe}")

    # Gather from multiple query variants + pagination
    queries = make_queries(recipe)
    all_urls, q_idx = [], 0
    while len(all_urls) < need and q_idx < len(queries):
        left = need - len(all_urls)
        urls = fetch_google_images(queries[q_idx], num_needed=left, max_pages=5)
        all_urls.extend(urls)
        q_idx += 1

    print(f"🔍 Collected {len(all_urls)} URLs for {recipe}")

    saved = 0
    start_idx = existing + 1        # <-- append after existing files
    for k, url in enumerate(all_urls, start=0):
        out_path = os.path.join(folder, f"{start_idx + saved:04d}.jpg")
        if download_image(url, out_path):
            saved += 1
        time.sleep(0.2)

    print(f"✅ Saved {saved} images for {recipe} (now total ~{existing + saved})")

print("\n🎉 All recipes processed successfully!")







# import os
# import requests
# import pandas as pd
# from dotenv import load_dotenv
# from PIL import Image
# from io import BytesIO
# import time

# # Load API key
# load_dotenv()
# SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# if not SERPAPI_KEY:
#     print("❌ ERROR: Missing SERPAPI_KEY in .env file!")
#     exit()

# # Load dataset
# try:
#     df = pd.read_excel("Punjabi food.xlsx")
# except FileNotFoundError:
#     print("❌ Error: 'Punjabi food.xlsx' not found.")
#     exit()

# # Create image folder
# base_dir = "Punjabi_Recipe_Images"
# os.makedirs(base_dir, exist_ok=True)

# # Utility functions
# def safe_name(name: str) -> str:
#     invalid_chars = '<>:"/\\|?*'
#     for ch in invalid_chars:
#         name = name.replace(ch, '')
#     return name.strip()

# def make_search_query(recipe_name: str) -> str:
#     return f"{recipe_name} Punjabi food dish recipe"

# def fetch_google_images(query: str, num_images: int = 25):
#     """Fetch image URLs using SerpApi Google Images"""
#     url = "https://serpapi.com/search.json"
#     params = {
#         "engine": "google",
#         "q": query,
#         "tbm": "isch",  # image mode
#         "num": num_images,
#         "api_key": SERPAPI_KEY
#     }

#     try:
#         response = requests.get(url, params=params, timeout=30)
#         data = response.json()
#         image_results = data.get("images_results", [])
#         return [img["original"] for img in image_results[:num_images]]
#     except Exception as e:
#         print(f"⚠️ Google Image fetch failed for '{query}': {e}")
#         return []

# def download_image(url, path):
#     try:
#         response = requests.get(url, timeout=15)
#         img = Image.open(BytesIO(response.content))
#         img.convert("RGB").save(path, "JPEG")
#         return True
#     except Exception:
#         return False

# # Main loop
# for recipe in df["Recipe Name"].dropna().unique():
#     folder_name = safe_name(recipe)
#     save_path = os.path.join(base_dir, folder_name)
#     os.makedirs(save_path, exist_ok=True)

#     # if len(os.listdir(save_path)) > 0:
#     #     print(f"✅ Already have images for: {recipe}")
#     #     continue
#     existing = len(os.listdir(save_path))
#     target = 200   # for example

#     if existing >= target:
#         print(f"✅ Already have {existing} images for {recipe}")
#         continue
#     else:
#         print(f"📥 Need {target-existing} more images for {recipe}")


#     query = make_search_query(recipe)
#     print(f"🔍 Searching for: {query}")

#     # urls = fetch_google_images(query, num_images=100)
#     urls = fetch_google_images(query, num_images=target-existing)
#     print(f"📸 Found {len(urls)} image URLs for {recipe}")

#     count = 0
#     for i, url in enumerate(urls, 1):
#         filename = os.path.join(save_path, f"{i}.jpg")
#         if download_image(url, filename):
#             count += 1
#         time.sleep(0.3)

#     if count == 0:
#         print(f"⚠️ No images downloaded for {recipe}")
#     else:
#         print(f"✅ Saved {count} images for {recipe}")

# print("\n🎉 All recipes processed successfully!")
