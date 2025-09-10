# recognize_plate.py

import os
import cv2
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import joblib
from utils import get_largest_rect_contour, get_ordered_rect_points, deskew, kmeans, split_char_by_kmeans_and_contours

# 模型路徑
PCA_MODEL_PATH = "models/tw_pca_model_v2.pkl"
GMM_MODELS_PATH = "models/tw_gmm_models_v2"
IMG_SIZE = (36, 18)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="img/MUY5686.jpg")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    pca, gmm_models = load_models()

    # 讀圖 + 預處理
    img = cv2.imread(args.image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edge = cv2.Canny(blur, 30, 200)

    rect = get_largest_rect_contour(edge)
    ordered = get_ordered_rect_points(rect)
    base_pts = np.array([[0, 0], [320, 0], [320, 150], [0, 150]], dtype="float32")
    plate = deskew(gray, base_pts, ordered)
    # cv2.imshow("Plate after deskew", plate)
    # cv2.waitKey(0)
    plate = plate[30:-20, :]  # 避開螺絲與邊緣

    _, plate = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate = cv2.bitwise_not(plate)
    cv2.imshow("Plate after bitwise_not", plate)
    cv2.waitKey(0)

    kernel = np.ones((3, 3), np.uint8)
    plate = cv2.erode(plate, kernel, iterations=1)
    plate = cv2.dilate(plate, kernel, iterations=1)

    ys, xs = np.where(plate == 255)
    coords = np.column_stack((xs, ys))

    centroids = kmeans(coords, K=7)
    char_imgs = split_char_by_kmeans_and_contours(centroids, plate)

    result = ""
    for idx, char in enumerate(char_imgs):
        if args.show:
            cv2.imshow(f"char_{idx+1}", char)
            cv2.waitKey(0)
        label = predict_character(char, pca, gmm_models)
        result += label

    cv2.destroyAllWindows()
    print("辨識結果：", result)

if __name__ == "__main__":
    main()
