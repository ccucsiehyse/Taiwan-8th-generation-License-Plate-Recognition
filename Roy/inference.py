import numpy as np
import cv2
import os

def inference(img):
    # 輸入圖片,輸出判斷的類別

    char = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N',
            'O','P','Q','R','S','T','U','V','W','X','Y','Z']
    # 載入線性模型參數
    W = np.load("weights.npy")
    b = np.load("biases.npy")

    logits = img @ W.T + b  # 單一樣本 x.shape = (648,)
    logits -= np.max(logits)  # 避免數值爆炸
    probs = np.exp(logits) / np.sum(np.exp(logits))
    predicted_class = np.argmax(probs)
    print(f"predicted_class = {predicted_class},char is {char[predicted_class]}")
    return char[predicted_class]

if __name__ == "__main__":
    # 測試
    img_height, img_width = 36,18 
    input_dir = r"D:\CNN_letter_Dataset\T"
    for file in os.listdir(input_dir):
        if file.endswith(".jpg"):
            path = os.path.join(input_dir, file)
            img = cv2.imread(path)
            cv2.imshow(f"origin",img)
            cv2.waitKey(0)
            img = cv2.resize(img, (img_width, img_height))
            cv2.imshow(f"resize",img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imshow(f"binary",binary)
            cv2.waitKey(0)
            x = binary.flatten().astype(np.float32)
            
            print(f"{file}:{inference(x)}")

