import cv2
import numpy as np

def get_largest_rect_contour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None  # 如果沒有找到符合條件的矩形輪廓，返回 None
    # raise Exception("找不到矩形輪廓")

def get_ordered_rect_points(contour):
    rect = np.zeros((4, 2), dtype="float32")
    s = contour.sum(axis=2)
    rect[0] = contour[np.argmin(s)]
    rect[2] = contour[np.argmax(s)]
    diff = np.diff(contour, axis=2)
    rect[1] = contour[np.argmin(diff)]
    rect[3] = contour[np.argmax(diff)]
    return rect

def deskew(src_img, base_rect, skewed_rect):
    M = cv2.getPerspectiveTransform(skewed_rect, base_rect)
    return cv2.warpPerspective(src_img, M, (int(base_rect[2][0]), int(base_rect[2][1])))

def kmeans(data, K, tol=1e-4):
    h = 100
    x_init = np.linspace(320/14, 320*13/14, K)
    centroids = np.array([[x, h/2] for x in x_init])
    for _ in range(20):
        dists = np.linalg.norm(data[:, None] - centroids, axis=2)
        labels = np.argmin(dists, axis=1)
        for k in range(K):
            points = data[labels == k]
            if len(points) > 0:
                centroids[k] = points.mean(axis=0)
    return centroids

def split_char_by_kmeans_and_contours(centroids, plate):
    # 先用 kmeans 分區，再用輪廓細調
    xs = np.sort([c[0] for c in centroids])
    split_points = [0]
    for i in range(1, len(xs)):
        split_points.append(int((xs[i-1] + xs[i]) / 2))
    split_points.append(plate.shape[1])

    chars = []
    for i in range(len(xs)):
        x1, x2 = split_points[i], split_points[i+1]
        x1, x2 = max(0, x1 - 10), min(x2 + 10, plate.shape[1]-1)  # 避免越界
        roi = plate[:, x1:x2]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            char = roi[y:y+h, x:x+w]
        else:
            char = roi
        chars.append(char)
    return chars
