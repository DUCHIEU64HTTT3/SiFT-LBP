import cv2
import os
import numpy as np
from tqdm import tqdm

# ================== PATH ==================
INPUT_DIR = r".\food-101\food-101\images"
OUTPUT_DIR = r".\food-101\food101_processed"

LBP_DIR = os.path.join(OUTPUT_DIR, "lbp")
SIFT_DIR = os.path.join(OUTPUT_DIR, "sift")

os.makedirs(LBP_DIR, exist_ok=True)
os.makedirs(SIFT_DIR, exist_ok=True)

categories = sorted(os.listdir(INPUT_DIR))

# ================== UTILS ==================
def read_image_safe(path):
    """Đọc ảnh an toàn với đường dẫn Unicode (Windows)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except:
        return None

# ================== PROCESS ==================
for category in tqdm(categories, desc="Tiền xử lý Food-101"):
    in_path = os.path.join(INPUT_DIR, category)
    if not os.path.isdir(in_path):
        continue

    out_lbp = os.path.join(LBP_DIR, category)
    out_sift = os.path.join(SIFT_DIR, category)
    os.makedirs(out_lbp, exist_ok=True)
    os.makedirs(out_sift, exist_ok=True)

    for fname in os.listdir(in_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(in_path, fname)
        img = read_image_safe(img_path)
        if img is None:
            print(f"⚠️ Không đọc được ảnh: {img_path}")
            continue

        try:
            # ---------- Resize ----------
            img = cv2.resize(img, (256, 256))

            # ---------- SIFT PIPELINE ----------
            sift_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sift_save = os.path.join(out_sift, fname)
            cv2.imencode('.jpg', sift_gray)[1].tofile(sift_save)

            # ---------- LBP PIPELINE ----------
            lbp_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp_blur = cv2.GaussianBlur(lbp_gray, (3,3), 0)
            lbp_eq = cv2.equalizeHist(lbp_blur)
            lbp_save = os.path.join(out_lbp, fname)
            cv2.imencode('.jpg', lbp_eq)[1].tofile(lbp_save)

        except Exception as e:
            print(f"❌ Lỗi xử lý {img_path}: {e}")

print("\n✅ Hoàn tất tiền xử lý Food-101")
print("📁 LBP images :", LBP_DIR)
print("📁 SIFT images:", SIFT_DIR)
