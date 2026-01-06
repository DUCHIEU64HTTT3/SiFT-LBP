import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm

# ===== CẤU HÌNH =====
DATASET_DIR = r".\food-101\food-101\images"  # thư mục chứa ảnh gốc Food-101
OUTPUT_DIR = "."                     # nơi lưu lbp_features.npy
IMG_SIZE = (128, 128)                # resize ảnh
RADIUS = 3
METHOD = 'uniform'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== HÀM HỖ TRỢ =====
def imread_unicode(path):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except:
        return None

def compute_lbp_histogram(gray_img, radius=RADIUS, method=METHOD):
    P = 8 * radius
    lbp = local_binary_pattern(gray_img, P, radius, method)
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # chuẩn hóa
    return hist.reshape(1, -1)

# ===== XỬ LÝ DATASET =====
features = []
labels = []
filenames = []

categories = sorted(os.listdir(DATASET_DIR))
for category in tqdm(categories, desc="Processing categories"):
    class_path = os.path.join(DATASET_DIR, category)
    if not os.path.isdir(class_path):
        continue
    
    for fname in os.listdir(class_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        fpath = os.path.join(class_path, fname)
        img = imread_unicode(fpath)
        if img is None:
            print(f"⚠️ Không đọc được ảnh: {fpath}")
            continue
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gray = cv2.equalizeHist(gray)
            gray = cv2.resize(gray, IMG_SIZE)
            hist = compute_lbp_histogram(gray)
            
            features.append(hist)
            labels.append(category)
            filenames.append(f"{category}/{fname}")
        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý ảnh {fpath}: {e}")

# ===== LƯU DỮ LIỆU =====
features = np.vstack(features).astype('float32')
labels = np.array(labels)
filenames = np.array(filenames)

np.save(os.path.join(OUTPUT_DIR, "lbp_features.npy"), features)
np.save(os.path.join(OUTPUT_DIR, "lbp_labels.npy"), labels)
np.save(os.path.join(OUTPUT_DIR, "lbp_filenames.npy"), filenames)

print(f"\n✅ Hoàn tất! Đã trích xuất LBP cho {len(filenames)} ảnh.")
print(f"Kích thước features: {features.shape}")
print(f"Số lớp món ăn: {len(set(labels))}")
