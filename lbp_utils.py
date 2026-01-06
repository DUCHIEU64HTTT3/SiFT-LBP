import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# --- Hàm đọc ảnh Unicode an toàn ---
def imread_unicode(path):
    """
    Đọc ảnh với đường dẫn Unicode (chứa ký tự tiếng Việt có dấu).
    """
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"⚠️ Lỗi đọc ảnh Unicode: {path} -> {e}")
        return None


def load_gray_image(path, resize=(128, 128)):
    """
    Đọc ảnh, chuyển sang ảnh xám và resize về kích thước chuẩn.
    """
    img = imread_unicode(path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if resize is not None:
        gray = cv2.resize(gray, resize)
    return gray


def compute_lbp_histogram(gray_img, radius=1, method='uniform'):
    """
    Tính đặc trưng LBP (Local Binary Pattern) cho ảnh xám và trả về histogram chuẩn hóa.
    """
    P = 8 * radius
    lbp = local_binary_pattern(gray_img, P, radius, method)
    n_bins = P + 2  # Đảm bảo giống bên Flask
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist.reshape(1, -1)
