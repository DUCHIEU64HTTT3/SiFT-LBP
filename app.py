import os
import re
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory, send_file
from skimage.feature import local_binary_pattern
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# ==================================================
# BASE PATH
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================================================
# CẤU HÌNH THƯ MỤC
# ==================================================
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MATCH_FOLDER  = os.path.join(BASE_DIR, "static", "sift_match")
LBP_FOLDER    = os.path.join(BASE_DIR, "static", "lbp_hist")

DATASET_PATH = os.path.join(BASE_DIR, "food-101", "food-101", "images")

TOP_K = 5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MATCH_FOLDER, exist_ok=True)
os.makedirs(LBP_FOLDER, exist_ok=True)

# ==================================================
# LOAD FEATURES
# ==================================================
lbp_features  = np.load(os.path.join(BASE_DIR, "lbp_features.npy"))
lbp_labels    = np.load(os.path.join(BASE_DIR, "lbp_labels.npy"))
lbp_filenames = np.load(os.path.join(BASE_DIR, "lbp_filenames.npy"))

SIFT_DIR = os.path.join(BASE_DIR, "food-101", "sift_processed")
sift_features  = np.load(os.path.join(SIFT_DIR, "sift_features.npy"))
sift_labels    = np.load(os.path.join(SIFT_DIR, "sift_labels.npy"))
sift_filenames = np.load(os.path.join(SIFT_DIR, "sift_filenames.npy"))

print("✅ Đã nạp dữ liệu LBP & SIFT")

# ==================================================
# FLASK INIT
# ==================================================
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# ==================================================
# HÀM ĐỌC ẢNH UNICODE (QUAN TRỌNG)
# ==================================================
def imread_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except:
        return None

# ==================================================
# HÀM HỖ TRỢ
# ==================================================
def sanitize_filename(name):
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)

# ==================================================
# LBP
# ==================================================
def compute_lbp_feature(img, radius=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (128,128))

    P = 8 * radius
    lbp = local_binary_pattern(gray, P, radius, method="uniform")
    n_bins = P + 2

    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, n_bins+1),
                           range=(0, n_bins))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)
    return hist

def chi_square_distance(a, b):
    return 0.5 * np.sum(((a - b)**2) / (a + b + 1e-10))

def search_lbp(query_hist):
    dists = np.array([chi_square_distance(f, query_hist) for f in lbp_features])
    idxs = np.argsort(dists)[:TOP_K]
    return [(lbp_filenames[i], lbp_labels[i], float(dists[i])) for i in idxs], dists

# ==================================================
# SIFT
# ==================================================
def compute_sift_feature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256,256))
    kp, des = sift.detectAndCompute(gray, None)
    if des is None:
        return np.zeros(128, dtype=np.float32)
    return des.mean(axis=0).astype(np.float32)

def search_sift(query_vec):
    dists = np.linalg.norm(sift_features - query_vec, axis=1)
    idxs = np.argsort(dists)[:TOP_K]
    return [(sift_filenames[i], sift_labels[i], float(dists[i])) for i in idxs], dists

# ==================================================
# ROUTES
# ==================================================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        method = request.form.get("method", "lbp")

        safe_name = sanitize_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(save_path)

        img = imread_unicode(save_path)
        if img is None:
            return render_template("index.html", error="Không đọc được ảnh (Unicode path)")

        lbp_img_name = None
        match_imgs = []

        # ----- LBP -----
        if method in ["lbp", "both"]:
            q_lbp = compute_lbp_feature(img)
            lbp_results, lbp_dists = search_lbp(q_lbp)

            plt.figure(figsize=(6,4))
            plt.bar(range(len(q_lbp)), q_lbp, color="orange")
            plt.title("LBP Histogram")
            plt.tight_layout()

            lbp_img_name = f"lbp_{os.path.splitext(safe_name)[0]}.png"
            plt.savefig(os.path.join(LBP_FOLDER, lbp_img_name))
            plt.close()

        # ----- SIFT -----
        if method in ["sift", "both"]:
            q_sift = compute_sift_feature(img)
            sift_results, sift_dists = search_sift(q_sift)

            query_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp1, des1 = sift.detectAndCompute(query_gray, None)

            for i, (fname, _, _) in enumerate(sift_results):
                best_path = os.path.join(DATASET_PATH, fname)
                best_img = imread_unicode(best_path)
                if best_img is None:
                                print("❌ Không đọc được ảnh dataset:", best_path)
                                continue
                best_gray = cv2.cvtColor(best_img, cv2.COLOR_BGR2GRAY)
                kp2, des2 = sift.detectAndCompute(best_gray, None)

                # ❌ Không có descriptor → BỎ QUA, KHÔNG LƯU
                if des1 is None or des2 is None:
                    print("⚠️ Không có descriptor:", fname)
                    continue

                matches = bf.knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]

                # ❌ Không có good match → BỎ QUA
                if len(good) == 0:
                    print("⚠️ Không có good match:", fname)
                    continue

                match_img = cv2.drawMatches(
                    query_gray, kp1,
                    best_gray, kp2,
                    good[:40],
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                base_name = os.path.splitext(safe_name)[0]
                out_name = f"sift_{i}_{base_name}.jpg"

                out_path = os.path.join(MATCH_FOLDER, out_name)

                success, encoded = cv2.imencode(".jpg", match_img)
                if success:
                        encoded.tofile(out_path)
                        print("✅ Đã lưu SIFT match (Unicode-safe):", out_path)
                        match_imgs.append(out_name)
                else:
                        print("❌ Không encode được ảnh:", out_path)


        # ----- FUSION -----
        if method == "both":
            lbp_n  = (lbp_dists  - lbp_dists.min())  / (np.ptp(lbp_dists)  + 1e-6)
            sift_n = (sift_dists - sift_dists.min()) / (np.ptp(sift_dists) + 1e-6)

            combined = 0.5 * lbp_n + 0.5 * sift_n
            idxs = np.argsort(combined)[:TOP_K]
            results = [(lbp_filenames[i], lbp_labels[i], float(combined[i])) for i in idxs]
        elif method == "lbp":
            results = lbp_results
        else:
            results = sift_results

        return render_template(
            "result.html",
            query=safe_name,
            results=results,
            lbp_img=lbp_img_name,
            match_imgs=match_imgs,
            method=method
        )

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/dataset/<path:filename>")
def dataset_image(filename):
    return send_file(os.path.join(DATASET_PATH, filename))


if __name__ == "__main__":
   app.run(debug=True, use_reloader=False)

