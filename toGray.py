import cv2
import matplotlib.pyplot as plt

# 讀取圖片
image_path = 'TW_plate_style.jpg'  # 請確保圖片路徑正確
img = cv2.imread(image_path)

# 檢查圖片是否正確讀取
if img is None:
    print("Error: Could not read image. Please check the file path.")
else:
    # 轉換為灰階圖像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化處理 (這裡使用 Otsu's 方法自動選擇閾值)
    # cv2.THRESH_BINARY 表示大於閾值的像素設為最大值，否則設為0
    # cv2.THRESH_OTSU 表示使用 Otsu's 方法自動計算最佳閾值
    ret, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print(f"Otsu's threshold: {ret}")

    # 顯示原始、灰階和二值化圖像
    plt.figure()
    # plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.figure()
    # plt.subplot(1, 3, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    plt.figure()
    # plt.subplot(1, 3, 3)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Binary Image (Otsu)')
    plt.axis('off')

    plt.show()

    # 也可以選擇保存處理後的圖片
    # cv2.imwrite('台灣車牌字形_grayscale.jpg', gray_img)
    cv2.imwrite('TW_plate_style.jpg', binary_img)