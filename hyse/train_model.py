# train_model.py

import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import joblib
from tqdm import tqdm

# 設定路徑與參數
dataset_path = r"D:\license_plate_datasets\TW_plate_digits"
MODEL_VERSION = "_demo"
output_model_path = f"models/tw_gmm_models{MODEL_VERSION}"
pca_model_path = f"models/tw_pca_model{MODEL_VERSION}.pkl"
n_components_pca = 50
n_components_gmm = 3  # 每類用 3 個高斯分布來擬合

# 建立儲存模型的資料夾
os.makedirs(output_model_path, exist_ok=True)

# 資料蒐集
X = []           # 所有圖像的特徵向量
y = []           # 對應的標籤
labels = sorted(os.listdir(dataset_path))  # ex: ['0', '1', ..., 'Z']，不含 O

print("開始讀取資料...")
for label in tqdm(labels):
    folder = os.path.join(dataset_path, label)
    for fname in os.listdir(folder):
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (36, 18)).flatten() / 255.0
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

# PCA 降維
pca = PCA(n_components=n_components_pca)
X_pca = pca.fit_transform(X)
joblib.dump(pca, pca_model_path)

# GMM 分類器訓練：每個類別一個 GMM
print("開始訓練 GMM 模型...")
for label in tqdm(labels):
    X_class = X_pca[y == label]
    gmm = GaussianMixture(n_components=n_components_gmm, covariance_type='full', random_state=42)
    gmm.fit(X_class)
    joblib.dump(gmm, os.path.join(output_model_path, f"{label}.pkl"))

print("模型訓練完成，PCA與GMM已儲存。")
