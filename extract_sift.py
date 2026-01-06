import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===== Đường dẫn =====
INPUT_DIR = r".\food-101\food-101\images"   # dataset gốc
OUTPUT_DIR = r".\food-101\sift_processed"   # thư mục lưu features
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Khởi tạo SIFT =====
sift = cv2.SIFT_create()

# ===== Lưu dữ liệu =====
features = []
labels = []
filenames = []

categories = sorted(os.listdir(INPUT_DIR))

# ======= TRÍCH ĐẶC TRƯNG SIFT =======
for category in tqdm(categories, desc="Processing categories"):
    category_path = os.path.join(INPUT_DIR, category)
    if not os.path.isdir(category_path):
        continue

    for fname in os.listdir(category_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(category_path, fname)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Không đọc được ảnh: {img_path}")
            continue

        img = cv2.resize(img, (256, 256))
        kp, des = sift.detectAndCompute(img, None)

        if des is None or len(des) == 0:
            vec = np.zeros(128, dtype=np.float32)
        else:
            vec = des.mean(axis=0)

        features.append(vec)
        labels.append(category)
        filenames.append(f"{category}/{fname}")

features = np.array(features, dtype=np.float32)
labels = np.array(labels)
filenames = np.array(filenames)

# ======= LƯU LẠI =======
np.save(os.path.join(OUTPUT_DIR, "sift_features.npy"), features)
np.save(os.path.join(OUTPUT_DIR, "sift_labels.npy"), labels)
np.save(os.path.join(OUTPUT_DIR, "sift_filenames.npy"), filenames)
print("✅ Hoàn tất trích đặc trưng! Features shape:", features.shape)

# ============================================================
# ======= BẮT ĐẦU PHẦN FLANN MATCH: TÌM ẢNH TƯƠNG TỰ =========
# ============================================================

# ---- 1️⃣ Ảnh cần tìm ----
query_path = r".\food-101\food-101\images\apple_pie\1005649.jpg"  # ảnh đầu vào
query_img = cv2.imdecode(np.fromfile(query_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
query_img = cv2.resize(query_img, (256, 256))
kp_q, des_q = sift.detectAndCompute(query_img, None)
if des_q is None:
    raise ValueError("Ảnh cần tìm không có đặc trưng SIFT nào!")
query_vec = des_q.mean(axis=0)

# ---- 2️⃣ Tính khoảng cách Euclidean giữa query và dataset ----
distances = np.linalg.norm(features - query_vec, axis=1)

# ---- 3️⃣ Lấy Top 5 ảnh có khoảng cách nhỏ nhất ----
top_indices = np.argsort(distances)[:5]
top_files = filenames[top_indices]
top_labels = labels[top_indices]
top_scores = distances[top_indices]

print("\n🖼️ Top 5 ảnh giống nhất:")
for i, (f, s) in enumerate(zip(top_files, top_scores), start=1):
    print(f"{i}. {f}  (Khoảng cách = {s:.4f})")

# ---- 4️⃣ Hiển thị ảnh đầu vào và 5 ảnh kết quả ----
plt.figure(figsize=(15, 6))
plt.subplot(2, 3, 1)
plt.imshow(query_img, cmap='gray')
plt.title("Ảnh đầu vào")
plt.axis('off')

for i, f in enumerate(top_files, start=2):
    img_path = os.path.join(INPUT_DIR, f)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    plt.subplot(2, 3, i)
    plt.imshow(img, cmap='gray')
    plt.title(f"Top {i-1}")
    plt.axis('off')

plt.tight_layout()
plt.show()
