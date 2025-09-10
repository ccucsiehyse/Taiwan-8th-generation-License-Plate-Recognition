# recognize_plate_v2.py
import os
import cv2
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from utils_v2 import sobel_based_plate_localization, kmeans, split_char_by_kmeans_and_contours

# 是否顯示預測錯誤的影像
SHOW_PREDICTIONS = 1

# 模型與影像設定
MODEL_VERSION = "_v2"
PCA_MODEL_PATH = f"models/tw_pca_model{MODEL_VERSION}.pkl"
GMM_MODELS_PATH = f"models/tw_gmm_models{MODEL_VERSION}"
IMG_SIZE = (36, 18)
TEST_DIR = "img"

# 預測單一字元
def predict_character(img, pca, gmm_models, allowed_labels=None):
    img = cv2.resize(img, IMG_SIZE).flatten() / 255.0
    img_pca = pca.transform([img])

    best_label = None
    best_score = -np.inf

    for label, gmm in gmm_models.items():
        if allowed_labels is not None and label not in allowed_labels:
            continue
        score = gmm.score(img_pca)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label

# 載入 PCA 與 GMM 模型
def load_models():
    pca = joblib.load(PCA_MODEL_PATH)
    gmm_models = {}
    for file in os.listdir(GMM_MODELS_PATH):
        if file.endswith(".pkl"):
            label = file.split(".")[0]
            gmm_models[label] = joblib.load(os.path.join(GMM_MODELS_PATH, file))
    return pca, gmm_models

# 辨識單一車牌影像
def recognize_plate(image_path, pca, gmm_models):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] 無法讀取圖片: {image_path}")
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plate, found = sobel_based_plate_localization(gray)
    if not found or plate is None or plate.size == 0:
        print(f"[Warning] 找不到車牌: {image_path}")
        return ""

    if plate.shape[0] > 50:
        plate = plate[30:-20, :]
    else:
        print(f"[Warning] 車牌高度太小: {image_path}")
        return ""

    # 二值化 + 反轉 + 形態學處理
    _, plate = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate = cv2.bitwise_not(plate)

    cv2.imshow(f"plate", plate)
    cv2.waitKey(0)

    kernel = np.ones((3, 3), np.uint8)
    plate = cv2.erode(plate, kernel, iterations=1)
    plate = cv2.dilate(plate, kernel, iterations=1)

    # 找出前景白色像素
    ys, xs = np.where(plate == 255)
    if len(xs) == 0:
        print(f"[Warning] 無法取得前景: {image_path}")
        return ""

    coords = np.column_stack((xs, ys))
    centroids = kmeans(coords, K=7)
    char_imgs = split_char_by_kmeans_and_contours(centroids, plate)

    result = ""
    for idx, char in enumerate(char_imgs):
        if idx < 3:
            allowed = [ch for ch in "ABCDEFGHJKLMNPQRSTUVWXYZ"]
        else:
            allowed = [str(d) for d in range(10)]

        label = predict_character(char, pca, gmm_models, allowed_labels=allowed)
        result += label

    return result

# 批次測試與統計
def evaluate():
    pca, gmm_models = load_models()

    total_chars = 0
    correct_chars = 0
    predicted_chars = 0
    total_plates = 0
    correct_plates = 0

    print("開始批次測試...")
    for fname in os.listdir(TEST_DIR):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        true_label = os.path.splitext(fname)[0]
        img_path = os.path.join(TEST_DIR, fname)
        pred_label = recognize_plate(img_path, pca, gmm_models)

        total_plates += 1
        total_chars += len(true_label)
        predicted_chars += len(pred_label)

        correct = sum(1 for a, b in zip(pred_label, true_label) if a == b)
        correct_chars += correct

        if pred_label == true_label:
            correct_plates += 1

        if SHOW_PREDICTIONS and pred_label != true_label:
            print(f"[錯誤] {fname} | 真實: {true_label} | 預測: {pred_label}")

    char_acc = correct_chars / predicted_chars if predicted_chars > 0 else 0
    coverage = predicted_chars / total_chars if total_chars > 0 else 0
    plate_acc = correct_plates / total_plates if total_plates > 0 else 0

    print("\n--- 評估結果 ---")
    print(f"字元準確率：{char_acc:.2%}")
    print(f"字元覆蓋率：{coverage:.2%}")
    print(f"完整車牌辨識率：{plate_acc:.2%}")

if __name__ == "__main__":
    evaluate()
