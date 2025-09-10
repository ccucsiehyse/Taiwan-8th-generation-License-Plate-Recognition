# recognize_plate.py（已整合前面版本 + 使用純 KMeans 分割）

import os
import cv2
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from utils_GMM import get_largest_rect_contour, get_ordered_rect_points, deskew, kmeans, split_char_by_kmeans

PCA_MODEL_PATH = "models/tw_pca_model_v2.pkl"
GMM_MODELS_PATH = "models/tw_gmm_models_v2"
IMG_SIZE = (36, 18)
TEST_DIR = "img"

def load_models():
    pca = joblib.load(PCA_MODEL_PATH)
    gmm_models = {}
    for file in os.listdir(GMM_MODELS_PATH):
        if file.endswith(".pkl"):
            label = file.split(".")[0]
            gmm_models[label] = joblib.load(os.path.join(GMM_MODELS_PATH, file))
    return pca, gmm_models

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

def recognize_plate(img_path, pca, gmm_models):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edge = cv2.Canny(blur, 30, 200)
    rect = get_largest_rect_contour(edge)
    if rect is None:
        return ""
    ordered = get_ordered_rect_points(rect)
    base_pts = np.array([[0, 0], [320, 0], [320, 150], [0, 150]], dtype="float32")
    plate = deskew(gray, base_pts, ordered)
    plate = plate[30:-20, :]
    _, plate = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate = cv2.bitwise_not(plate)
    kernel = np.ones((3, 3), np.uint8)
    plate = cv2.erode(plate, kernel, iterations=1)
    plate = cv2.dilate(plate, kernel, iterations=1)

    ys, xs = np.where(plate == 255)
    coords = np.column_stack((xs, ys))
    if len(coords) == 0:
        return ""

    centroids = kmeans(coords, K=7)
    char_imgs = split_char_by_kmeans(centroids, plate)

    result = ""
    for char in char_imgs:
        label = predict_character(char, pca, gmm_models)
        result += label
    return result

def evaluate():
    pca, gmm_models = load_models()
    total_chars = 0
    correct_chars = 0
    total_images = 0
    correct_plates = 0
    total_predicted_chars = 0

    for filename in os.listdir(TEST_DIR):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        true_label = os.path.splitext(filename)[0].upper()
        img_path = os.path.join(TEST_DIR, filename)
        pred_label = recognize_plate(img_path, pca, gmm_models)

        print(f"[{filename}] Truth: {true_label} | Predict: {pred_label}")

        total_images += 1
        total_chars += len(true_label)
        total_predicted_chars += len(pred_label)

        for t_char, p_char in zip(true_label, pred_label):
            if t_char == p_char:
                correct_chars += 1

        if pred_label == true_label:
            correct_plates += 1

    char_acc = correct_chars / total_chars if total_chars else 0
    coverage = total_predicted_chars / total_chars if total_chars else 0
    plate_acc = correct_plates / total_images if total_images else 0

    print("\n=== evaluate result ===")
    print(f"character Accuracy : {char_acc:.2%}")
    print(f"character Coverage: {coverage:.2%}")
    print(f"    plate Accuracy: {plate_acc:.2%}")

if __name__ == "__main__":
    evaluate()
