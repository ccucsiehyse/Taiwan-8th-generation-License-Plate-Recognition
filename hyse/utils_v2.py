# utils_v2.py
import cv2
import numpy as np
from sklearn.cluster import KMeans

def sobel_based_plate_localization(gray):
    """
    使用 Sobel 垂直邊緣偵測與垂直投影來定位車牌區域。
    傳入灰階圖，回傳切下來的 plate 區域（灰階）、是否成功。
    """
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)
    _, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 垂直投影（對每一列計算白色像素數）
    projection = np.sum(binary, axis=1)
    max_val = np.max(projection)
    threshold = max_val * 0.3

    upper = 0
    lower = gray.shape[0] - 1
    for i in range(len(projection)):
        if projection[i] > threshold:
            upper = i
            break
    for i in reversed(range(len(projection))):
        if projection[i] > threshold:
            lower = i
            break

    if lower - upper < 15:
        return None, False

    plate_region = gray[upper:lower, :]
    return plate_region, True

def kmeans(data, K=7):
    """
    對座標資料進行 KMeans 分群。
    """
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_[:, 0]  # 取 X 座標中心
    return np.sort(centers)

def split_char_by_kmeans_and_contours(xs, binary_plate):
    """
    根據 kmeans 分群出的 x 軸中心點與輪廓，切割出字元影像。
    回傳 list of 影像。
    """
    char_images = []
    sorted_xs = sorted(xs)
    plate_h, plate_w = binary_plate.shape

    for i in range(len(sorted_xs)):
        cx = int(sorted_xs[i])

        # 每個中心點為中心，切一個固定寬度的區域
        left = max(cx - 10, 0)
        right = min(cx + 10, plate_w)
        char = binary_plate[:, left:right]

        # 濾除過小的片段
        if char.shape[1] < 5:
            continue

        # 加邊框補成固定大小輸出
        char = cv2.resize(char, (18, 36), interpolation=cv2.INTER_NEAREST)
        char_images.append(char)

    return char_images
