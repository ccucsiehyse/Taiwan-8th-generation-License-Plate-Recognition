import cv2
import numpy as np

# 全域變數
drawing = False # True if mouse is pressed
ix, iy = -1, -1 # Starting x,y coordinates
img_original = None # To store the original image
img_display = None # To display the image with current rectangle

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img_display, img_original

    # 複製一份原始圖片到 img_display，以便每次繪製都從乾淨的圖片開始
    img_display = img_original.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            # 實時繪製矩形，顯示當前選取範圍
            cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2) # 綠色矩形

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 繪製最終矩形
        cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)

        # 確保選擇的區域有效 (左上角 < 右下角)
        x_min = min(ix, x)
        y_min = min(iy, y)
        x_max = max(ix, x)
        y_max = max(iy, y)

        if x_max - x_min > 0 and y_max - y_min > 0:
            # 截取原始圖片中的區域
            cropped_image = img_original[y_min:y_max, x_min:x_max]

            # 保存截取後的圖片
            # 可以根據需要修改保存的路徑和檔名
            output_filename = f"TW_plate_digits/{x_min}_{y_min}.jpg"
            cv2.imwrite(output_filename, cropped_image)
            print(f"區域已保存為: {output_filename}")
        else:
            print("選擇的區域太小或無效，請重新選擇。")

def main():
    global img_original, img_display

    image_path = 'TW_plate_style.jpg'  # 請將此處替換為你的圖片檔案路徑
    img_original = cv2.imread(image_path)

    if img_original is None:
        print(f"錯誤：無法讀取圖片 '{image_path}'。請檢查路徑和檔名。")
        return

    img_display = img_original.copy() # 初始化顯示圖片

    cv2.namedWindow('Image Cropper')
    cv2.setMouseCallback('Image Cropper', draw_rectangle)

    print("請使用滑鼠左鍵拖曳選取區域。按 'q' 鍵退出。")

    while True:
        cv2.imshow('Image Cropper', img_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()