import cv2
import numpy as np
import os
import random
from tqdm import tqdm # 用於顯示進度條

def augment_image(image):
    """
    對單張圖片進行隨機數據增強。
    輸入:
        image: 灰度圖像 (NumPy array)
    輸出:
        augmented_img: 增強後的灰度圖像 (NumPy array)
    """
    rows, cols = image.shape[:2]

    # 設定白色填充值 (適用於灰度圖像)
    white_fill_value = 255
    # 設定黑色填充值 (用於特定情況，如縮放導致裁剪時)
    black_fill_value = 0

    # 複製圖像，避免修改原始圖像
    processed_img = image.copy()

    # --- 1. 隨機縮放 (Random Scaling) ---
    scale_factor = random.uniform(0.95, 1.00) # 縮放範圍 0.85 到 1.15
    scaled_img = cv2.resize(processed_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # 將縮放後的圖片置於原始圖片大小的中心，避免資訊丟失
    new_rows, new_cols = scaled_img.shape[:2]

    # 創建一個與原始圖片大小相同的畫布，填充為白色
    canvas = np.full((rows, cols), white_fill_value, dtype=image.dtype)

    # 計算將 scaled_img 放置到 canvas 中心的起始座標
    # 處理縮放後比原圖小的情況
    if new_rows < rows:
        canvas_start_row = (rows - new_rows) // 2
        scaled_img_rows_to_copy = new_rows
    else: # 縮放後比原圖大，需要裁剪
        canvas_start_row = 0
        scaled_img_rows_to_copy = rows
        scaled_img_start_row = (new_rows - rows) // 2
        scaled_img = scaled_img[scaled_img_start_row : scaled_img_start_row + rows, :]

    if new_cols < cols:
        canvas_start_col = (cols - new_cols) // 2
        scaled_img_cols_to_copy = new_cols
    else: # 縮放後比原圖大，需要裁剪
        canvas_start_col = 0
        scaled_img_cols_to_copy = cols
        scaled_img_start_col = (new_cols - cols) // 2
        scaled_img = scaled_img[:, scaled_img_start_col : scaled_img_start_col + cols]

    # 將處理後的 scaled_img 放置到 canvas 上
    canvas[canvas_start_row : canvas_start_row + scaled_img_rows_to_copy,
           canvas_start_col : canvas_start_col + scaled_img_cols_to_copy] = scaled_img
    
    processed_img = canvas


    # # --- 2. 隨機平移 (Random Translation) ---
    # tx = random.randint(-cols // 16, cols // 16) # 左右平移最多 1/16 寬度
    # ty = random.randint(-rows // 16, rows // 16) # 上下平移最多 1/16 高度
    # M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    # processed_img = cv2.warpAffine(processed_img, M_trans, (cols, rows), borderValue=white_fill_value) # 邊界填充白色


    # --- 3. 隨機旋轉 (Random Rotation) ---
    angle = random.randint(-6, 6) # 旋轉角度 -6 到 6 度
    M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    processed_img = cv2.warpAffine(processed_img, M_rot, (cols, rows), borderValue=white_fill_value) # 邊界填充白色


    # # --- 4. 隨機亮度調整 (Random Brightness) ---
    # brightness_factor = random.uniform(0.6, 1.4) # 亮度調整範圍 0.6 到 1.4
    # processed_img = np.clip(processed_img * brightness_factor, 0, 255).astype(np.uint8)


    # --- 5. 隨機噪音 (Random Noise - Gaussian Noise) ---
    if random.random() < 0.2: # 20% 機率添加噪音
        mean = 0
        var = random.uniform(10, 50) # 噪音方差
        sigma = var**0.5
        
        # 確保噪音維度與圖像相同，對於灰度圖像，它是 (rows, cols)
        # 生成浮點數噪音，這樣可以有負值
        gauss = np.random.normal(mean, sigma, processed_img.shape).astype(np.float32) 
        
        # 將圖像轉換為浮點數類型，以便與噪音進行數學運算
        processed_img = processed_img.astype(np.float32)
        
        # 圖像和噪音相加
        processed_img = cv2.add(processed_img, gauss) 
        
        # 裁剪到 0-255 範圍並轉換回 np.uint8
        processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)

    # # --- 6. 隨機變形 (Random Shear) ---
    # # 這裡我們只做單向的剪切變形，更複雜的可以考慮透視變換
    # if random.random() < 0.2: # 20% 機率進行變形
    #     shear_factor_x = random.uniform(-0.1, 0.1) # X 軸剪切
    #     shear_factor_y = random.uniform(-0.05, 0.05) # Y 軸剪切 (通常車牌字元Y軸剪切較小)

    #     # 為了更穩定的剪切，可以考慮定義變換點
    #     # 這裡使用簡單的仿射變換來模擬剪切
    #     pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
    #     pts2 = np.float32([[0 + cols * shear_factor_y, 0], # Y 軸對 X 軸的剪切
    #                        [cols-1, 0 + cols * shear_factor_x], # X 軸對 Y 軸的剪切
    #                        [0, rows-1]])

    #     # 簡化剪切，只影響一個軸向，避免過度複雜變形
    #     # 例如，只做水平剪切
    #     # pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    #     # pts2 = np.float32([[0, 0 + rows * shear_factor_y], [cols - 1, 0], [0, rows - 1 + rows * shear_factor_y]])
    #     # 這裡會調整點2和點3，使其產生水平剪切效果

    #     # 重新定義仿射變換的點，確保合理剪切
    #     # 這是更常用的剪切矩陣方法，直接構建一個剪切矩陣
    #     M_shear = np.float32([
    #         [1, shear_factor_x, 0],
    #         [shear_factor_y, 1, 0]
    #     ])
        
    #     # 應用仿射變換，確保圖像中心點在變換中心
    #     M_shear[0, 2] = -shear_factor_x * cols / 2
    #     M_shear[1, 2] = -shear_factor_y * rows / 2

    #     processed_img = cv2.warpAffine(processed_img, M_shear, (cols, rows), borderValue=white_fill_value)

    return processed_img

def process_character_directory(char_dir_path, num_augmentations=1000):
    """
    處理單個字元資料夾，對其中的唯一圖片進行數據增強。
    """
    image_files = [f for f in os.listdir(char_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print(f"警告：資料夾 '{char_dir_path}' 中沒有找到圖片。跳過。")
        return

    # 假設每個資料夾只有一張原始圖片
    original_image_path = os.path.join(char_dir_path, image_files[0])
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE) # 以灰度模式讀取

    if original_image is None:
        print(f"錯誤：無法讀取圖片 '{original_image_path}'。跳過。")
        return

    print(f"正在為 '{image_files[0]}' (在 '{char_dir_path}') 生成 {num_augmentations} 張增強圖片...")

    for i in tqdm(range(num_augmentations)):
        augmented_img = augment_image(original_image.copy()) # 傳入副本，避免原始圖片被修改
        output_filepath = os.path.join(char_dir_path, f"{os.path.basename(image_files[0]).split('.')[0]}_aug_{i+1:04d}.jpg")
        cv2.imwrite(output_filepath, augmented_img)

def main():
    base_dir = 'TW_plate_digits' # 你的基礎資料夾名稱

    if not os.path.exists(base_dir):
        print(f"錯誤：基礎資料夾 '{base_dir}' 不存在。請確認路徑。")
        print("請確保你的資料夾結構如下：")
        print("TW_plate_digits/")
        print("├── A/")
        print("│   └── A.jpg")
        print("├── B/")
        print("│   └── B.jpg")
        print("└── 0/")
        print("    └── 0.jpg")
        return

    # 遍歷基礎資料夾下的所有子資料夾
    for char_name in os.listdir(base_dir):
        char_dir_path = os.path.join(base_dir, char_name)

        if os.path.isdir(char_dir_path):
            process_character_directory(char_dir_path, num_augmentations=1000)

    print("\n所有字元的數據增強已完成！")

if __name__ == '__main__':
    main()