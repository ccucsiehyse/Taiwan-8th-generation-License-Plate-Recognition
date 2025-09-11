import numpy as np
import cv2
import os

# 圖片大小設定
img_height, img_width = 36, 18
feature_size = img_height * img_width
num_classes = 36  # 0~9 + A~Z

# 資料路徑
data_dir = r"D:\CNN_letter_Dataset"

# 類別轉整數對應表：0~9 -> 0~9，A~Z -> 10~35
all_classes = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}


# 資料儲存區
data = []
labels = []

# 掃描每個子資料夾（類別）
for cls in sorted(os.listdir(data_dir)):
    cls_path = os.path.join(data_dir, cls)
    if not os.path.isdir(cls_path) or cls not in class_to_idx:
        print("if not os.path.isdir(cls_path) or cls not in class_to_idx")
        continue
    cls_idx = class_to_idx[cls]

    for it,file in enumerate(os.listdir(cls_path)):
        if not file.lower().endswith((".jpg", ".png", ".bmp")):
            continue
        path = os.path.join(cls_path, file)
        img = cv2.imread(path)
        if img is None:
            print(f"img {cls}/{file} is None")
            continue

        img = cv2.resize(img, (img_width, img_height))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = 255 - binary

        if(it == 1000): # 檢查二值化結果
            cv2.imshow(f"binary",binary)
            cv2.waitKey(0)
        x = binary.flatten().astype(np.float32)
        data.append(x)
        labels.append(cls_idx)
    print(f"folder {cls} finished")

# 轉為 numpy array
data = np.array(data)
labels = np.array(labels)
N = len(data)  # 總樣本數
print(f"Total training samples: {N}")

# 初始化參數 W 和 b
W = np.random.randn(num_classes, feature_size) * 0.001  # 隨機初始化,shape: (36, 648)
b = np.ones((num_classes,))               # shape: (36,)

max_iteration = 300 # 訓練次數
learning_rate = 0.0005
loss_history=[]
for iter in range(max_iteration):
    # 前向傳播：計算 logits 和 softmax
    logits = data @ W.T + b  #  [36,648] * [648,36] + [36] = [36 36]
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # 減去最大值，避免指數數值過大導致溢位
    
    # softmax p_i = exp_logits_i/sum(exp_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # shape: (N, 36)

    # Cross-entropy loss
    eps = 1e-10
    log_probs = -np.log(probs[np.arange(N), labels] + eps) # 加上 eps 以避免 log0
    ave_loss = np.mean(log_probs)
    loss_history.append(ave_loss)

    # 計算梯度
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(N), labels] = 1  # 將真實標籤轉換為 one-hot

    # 導數 dL/dy_i
    delta = (probs - y_onehot) / N  # shape: (N, 36)

    # 反向傳播：計算 W 和 b 的梯度
    dW = delta.T @ data  # shape: (36, 648)
    db = np.sum(delta, axis=0)  # shape: (36,)

    # 更新參數
    W -= learning_rate * dW
    b -= learning_rate * db
    print(f"epoch {iter} finished,Loss = {ave_loss}")


# 儲存參數
np.save("weights.npy",W)
np.save("biases.npy", b)

