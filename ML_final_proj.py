import os
import cv2
import numpy as np
import time
import argparse
import inference

def get_largest_rect_contour(image):
    # 找出面積最大的矩形輪廓
    contours, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    rect_contour = None
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)
        if len(approx) == 4:
            rect_contour = approx
            break

    if rect_contour is None:
        raise Exception("No rect found")
    return rect_contour


def get_ordered_rect_points(contour):
    #  左下,右下,右上,左上
    # the order is, bottom-left, bottom-right, top-right, top-left
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
    dest_img = cv2.warpPerspective(
        src_img, M, (int(base_rect[2][0]), int(base_rect[2][1])))
    return dest_img


# def get_n_largest_components(image, n):
#     # 使用 OpenCV 找出所有連通元件（connected components）：
#     # num_labels：總標籤數（含背景）
#     # labels：同樣大小的影像，每個 pixel 存的是其所屬的標籤編號（0 為背景）
#     # stats：每個元件的統計資訊（x, y, 寬, 高, 面積）
#     # connectivity=8：以 8 鄰域方式連接
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
#         image, connectivity=8)

#     areas = stats[:, cv2.CC_STAT_AREA]
#     sorted_areas_indices = np.argsort(-areas)

#     component_images = []
#     component_image_x = []

#     for i in range(1, n + 1):
#         largest_component = sorted_areas_indices[i]

#         x = stats[largest_component, cv2.CC_STAT_LEFT]
#         y = stats[largest_component, cv2.CC_STAT_TOP]
#         width = stats[largest_component, cv2.CC_STAT_WIDTH]
#         height = stats[largest_component, cv2.CC_STAT_HEIGHT]

#         component_image = np.zeros_like(image)
#         component_image[labels == largest_component] = 255
#         component_image = component_image[y:y+height, x:x+width]
#         component_images.append(component_image)
#         component_image_x.append(x)

#     # Sort the components by x position
#     component_images = [x for _, x in sorted(
#         zip(component_image_x, component_images))]

#     return component_images


# def get_templates(dir_path):
#     templates = {}
#     for file in os.listdir(dir_path):
#         if file.endswith(".jpg"):
#             template_image = cv2.imread(os.path.join(dir_path, file), 0)
#             filename = os.path.splitext(file)[0]
#             templates[filename] = template_image
#     # key: filename, value, template_image
#     return templates


def get_plate_str(images, templates, logs=False):
    plate_str = ""
    for parts in images:
        best_match = None
        best_match_value = 0
        for key, value in templates.items():
            result = cv2.matchTemplate(parts, value, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if best_match is None or max_val > best_match_value:
                best_match = key
                best_match_value = max_val
        plate_str += best_match
        if logs:
            print(f"best match: {best_match}, value: {best_match_value}")
    return plate_str


def kmeans(data,K,tol=1e-4):
    # 讀取數據
    max_iteration = 20

    # 初始中心點   
    width = 320
    height = 100
    x_positions = np.linspace(width/14,(width/14)*13, K) # 平均分散在0~320(寬)之間
    centroids = np.array([[x, height/2] for x in x_positions])


    distortion_history = []

    for iter in range(max_iteration):
        if iter > 2:
            diff = abs(distortion_history[-1] - distortion_history[-2])
            if diff/distortion_history[-2] < tol:
                break # distortion不再下降，提前結束

        # E-step: 分配點到最近的簇
        n_samples = data.shape[0] # data數量
        distances = np.empty((n_samples, K)) # 初始化陣列

        for i in range(n_samples):
            for k in range(K):
                # 計算第i個點到第k個中心的歐氏距離
                distances[i, k] = np.sqrt(np.sum((data[i] - centroids[k])**2))
            #     print(f"point {i+1} cen {k}: {distances[i, k]:.2f}") # 所有顯示距離
            # print("\n")
                
        labels = np.argmin(distances, axis=1) # 對每個row找到數值最小的index

        # 計算distortion function (E-step後)
        current_distortion = np.sum(np.min(distances, axis=1)) # 分配資料點給K個中心裡計算距離最短的，
        distortion_history.append(current_distortion)

        
        # M-step: 更新簇中心
        new_centroids = np.empty((K, data.shape[1]))  # 初始化新中心点数组
        for k in range(K):
            cluster_points = data[labels == k] # 獲取中心點為k的資料點
            
            # 如果簇不为空，计算均值；否则保持原中心点
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0) # 分別計算x,y,z的平均
            else:
                new_centroids[k] = centroids[k]  # 防止空簇
        centroids = new_centroids
        
        # 記錄M-step後的失真函數值
        distances = np.sqrt(((data[:, np.newaxis] - centroids)**2).sum(axis=2))
        current_distortion = np.sum(np.min(distances, axis=1))
        distortion_history.append(current_distortion)

        for i in range(n_samples):
            for k in range(K):
                # 計算第i個點到第k個中心的歐氏距離
                distances[i, k] = np.sqrt(np.sum((data[i] - centroids[k])**2))
                # print(f"point {i+1} cen {k}: {distances[i, k]:.2f}") # 所有顯示距離
            # print(f"\n")

        labels = np.argmin(distances, axis=1) # 對每個row找到數值最小的index
    return centroids
def split_char(centroids,plate):
    # 取得所有 x 座標，並排序（以防順序錯誤）
    xs = np.sort([c[0] for c in centroids])
    xs = np.array(xs)

    # 計算分界點（總共 K+1 個界線）
    split_points = []

    # 最左界：從圖像起始到第一個中心點左側的一半
    split_points.append(0)
    for i in range(1, len(xs)):
        mid = (xs[i - 1] + xs[i]) / 2
        split_points.append(int(mid))
    # 最右界：最後一個中心點右側延伸到圖像結尾
    split_points.append(320)  # 寬度

    # 切出每個字元區塊
    chars = []
    for i in range(len(xs)):
        x_start = split_points[i]
        x_end = split_points[i + 1]
        char_img = plate[:, x_start:x_end]
        chars.append(char_img)

    return chars
def main():

    # step1:圖像預處理
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str,
                        help="path to image file", default="img/plate1.jpg")
    parser.add_argument("--template_path", type=str,
                        help="path to template folder", default="template")
    parser.add_argument("--show_image", type=bool,
                        help="show image, Value: True or False", default=False)

    img = cv2.imread(parser.parse_args().image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    edge = cv2.Canny(filtered, 30, 200)
    # cv2.imshow('plate of edge', edge)

    # 找出面積最大的封閉矩形輪廓
    rect_contour = get_largest_rect_contour(edge)

    # 左下,右下,右上,左上
    ordered_rect_points = get_ordered_rect_points(rect_contour)

    # 建立一個矩形的基準點座標（左上、右上、右下、左下),做為透視變換的目標點
    plate_size_match = (int(320), int(150))
    base_plate_points = np.array([[0, 0], [plate_size_match[0], 0], [plate_size_match[0], plate_size_match[1]], [
        0, plate_size_match[1]]], dtype="float32")

    # 透視變換
    plate = deskew(gray, base_plate_points, ordered_rect_points)
    # cv2.imshow('plate after deskew', plate)

    # 裁切上下部分，避開上半螺絲區域和下方空白
    plate = plate[30:-20, :]

    
    # 將像素值轉換成0或255
    _, plate = cv2.threshold(
        plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 反轉顏色(文字變成白色，背景為黑色)
    plate = cv2.bitwise_not(plate)

    kernel = np.ones((3, 3), np.uint8)
    plate = cv2.erode(plate, kernel, iterations=1) # 侵蝕，讓白色變小
    plate = cv2.dilate(plate, kernel, iterations=1) # 膨脹，讓白色變粗

    # cv2.imshow('plate after preprocess', plate)
    # cv2.imwrite('plate after preprocess.jpg', plate)
    # cv2.waitKey(0)
    # =====================================================================
    # setp2:kmeans 分割圖片
    print("now perform kmeans...")

    # 將白色像素位置轉為 KMeans 輸入資料
    ys, xs = np.where(plate == 255)
    data = np.column_stack((xs, ys))  # 每列是一個 [x, y]
    
    centroids = kmeans(data,7,)
    print(centroids)

    # 取得分割好的7個圖片
    divided_image = split_char(centroids,plate)
    # for idx, img in enumerate(divided_image):
    #     cv2.imshow(f'divided_image_{idx+1}', img)  # 使用不同的視窗名稱
    #     cv2.imwrite(f'divided_image_{idx+1}.jpg', img)
    #     cv2.waitKey(0)

    # 開啟樣本圖以及對應表
    img_height, img_width = 36,18 
    result = ""
    for idx, img in enumerate(divided_image):
        img = cv2.resize(img, (img_width, img_height))

        cv2.imshow(f'resize_divided_image_{idx+1}', img)
        cv2.waitKey(0)

        img = img.flatten().astype(np.float32) # 拉申成一維陣列
        result = result + (inference.inference(img))
    
    print(f"plate is {result}")


if __name__ == "__main__":
    main()
