import os
import cv2
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from utils import get_largest_rect_contour, get_ordered_rect_points, deskew, kmeans, split_char_by_kmeans_and_contours

SHOW_PREDICTIONS = 0  # 是否顯示預測結果(錯誤)

# 模型路徑
MODEL_VERSION = "_v3"
PCA_MODEL_PATH = f"models/tw_pca_model{MODEL_VERSION}.pkl"
GMM_MODELS_PATH = f"models/tw_gmm_models{MODEL_VERSION}"
IMG_SIZE = (36, 18)
TEST_DIR = "img"

def predict_character(img, pca, gmm_models):
    img = cv2.resize(img, IMG_SIZE).flatten() / 255.0
    img_pca = pca.transform([img])

    best_label = None
    best_score = -np.inf

    for label, gmm in gmm_models.items():
        score = gmm.score(img_pca)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label

def load_models():
    pca = joblib.load(PCA_MODEL_PATH)
    gmm_models = {}
    for file in os.listdir(GMM_MODELS_PATH):
        if file.endswith(".pkl"):
            label = file.split(".")[0]
            gmm_models[label] = joblib.load(os.path.join(GMM_MODELS_PATH, file))
    return pca, gmm_models

def recognize_plate(image_path, pca, gmm_models):
    img = cv2.imread(image_path)
    if img is None:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edge = cv2.Canny(blur, 30, 200)

    rect = get_largest_rect_contour(edge)
    if rect is None or len(rect) < 4:
        return ""
    
    ordered = get_ordered_rect_points(rect)
    base_pts = np.array([[0, 0], [320, 0], [320, 150], [0, 150]], dtype="float32")
    plate = deskew(gray, base_pts, ordered)
    if plate is None:
        cv2.imshow(f"plate", plate)
        cv2.waitKey(0)
    plate = plate[30:-20, :]  # 去除邊緣與螺絲
    record_plate = plate.copy()

    _, plate = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate = cv2.bitwise_not(plate)

    kernel = np.ones((3, 3), np.uint8)
    plate = cv2.erode(plate, kernel, iterations=1)
    plate = cv2.dilate(plate, kernel, iterations=1)

    ys, xs = np.where(plate == 255)
    if len(xs) == 0:
        return ""

    coords = np.column_stack((xs, ys))
    centroids = kmeans(coords, K=7)
    char_imgs = split_char_by_kmeans_and_contours(centroids, plate)

    result = ""
    for idx, char in enumerate(char_imgs):
        label = predict_character(char, pca, gmm_models)
        result += label
    
    # print(f"辨識結果：{result} (長度: {len(result)})")
    return result

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

        print(f"[{fname}] True: {true_label} | Predict: {pred_label}")

    char_acc = correct_chars / predicted_chars if predicted_chars > 0 else 0
    coverage = predicted_chars / total_chars if total_chars > 0 else 0
    plate_acc = correct_plates / total_plates if total_plates > 0 else 0

    print("\n=== evaluate result ===")
    print(f"character Accuracy : {char_acc:.2%}")
    print(f"character Coverage: {coverage:.2%}")
    print(f"    plate Accuracy: {plate_acc:.2%}")

if __name__ == "__main__":
    evaluate()
