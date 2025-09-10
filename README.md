# 台灣車牌辨識系統 (Taiwan License Plate Recognition)

這是在「機器學習概論」課程中開發的期末專案。本專案旨在應用課堂所學的 **Clustering**, **Feature Extraction**, **Classification** 等機器學習技術，從頭開始建立一個可行的台灣車牌辨識 (LPR) 系統。

專案的核心特色是透過多次 **迭代改進 (Iterative Improvement)**，從最初 **26.1%** 的辨識率，逐步將系統性能提升至最終的 **78.10%**。

## 專案特色

  - **多版本迭代**：從基礎的線性分類器逐步演進到 PCA + GMM 模型，記錄了完整的優化歷程。
  - **台灣車牌特化**：針對台灣車牌的特定格式（前3英文字母 + 後4數字）進行規則限制與優化。
  - **完整工具鏈**：提供了從資料收集、圖像預處理、資料增強到模型訓練與評估的完整腳本。
  - **詳細實驗記錄**：儲存了不同開發階段的模型與測試結果，便於追蹤與比較。

## 技術流程

系統的辨識流程主要分為以下五個步驟：

1.  **圖像預處理 (Image Preprocessing)**：

      - 轉換為灰階 (Grayscale)
      - 雙邊濾波 (Bilateral Filter) 以減少噪點同時保留邊緣
      - Canny 邊緣檢測 (Canny Edge Detection)
      - 圖像裁剪與透視矯正 (Cropping & Deskew)
      - 腐蝕與膨脹 (Erosion & Dilation)

2.  **字元分割 (Character Segmentation)**：

      - 利用 **K-means 演算法** 將車牌上的白色像素點分群，初步定位七個字元的位置。
      - 結合輪廓分析 (Contour Analysis) 尋找最大輪廓面積，以提高分割的準確性。

3.  **特徵提取 (Feature Extraction)**：

      - 將分割後的字元圖像大小調整為 (36, 18)。
      - 使用 **主成分分析 (Principal Component Analysis, PCA)** 將 648 維的特徵向量降至 50 維，以捕捉關鍵特徵並減少計算量。

4.  **字元分類 (Character Classification)**：

      - 使用 **高斯混合模型 (Gaussian Mixture Model, GMM)** 對降維後的特徵進行擬合與分類。每個類別（A-Z, 0-9）都由 3 個高斯分量組成。

5.  **後處理 (Post-processing)**：

      - 應用台灣車牌的規則限制（前3碼為英文字母，後4碼為數字），過濾掉不合理的預測結果，大幅提升辨識率。

## 迭代改進與辨識率提升

本專案的關鍵在於不斷發現問題並透過迭代解決，以下是辨識率的提升歷程：

| 階段 | 主要方法 | 辨識率 | 關鍵改進策略 |
| :-- | :-- | :--: | :-- |
| **初始狀態** | 線性分類器 + Kaggle 資料集 | **26.1%** | 建立基礎辨識流程。 |
| **改進 1** | 線性分類器 + 自建台灣車牌資料集 | **28.6%** | **解決資料不匹配問題**：發現 Kaggle 資料集與台灣車牌樣式不同，因此自行裁切並透過資料增強建立專屬資料集。 |
| **改進 2** | **PCA + GMM** + 自建資料集 | **34.3%** | **升級分類模型**：意識到線性模型無法處理複雜邊界，改用 PCA 降維搭配 GMM 進行非線性特徵擬合。 |
| **改進 3** | PCA + GMM + **優化字元分割** | **54.76%** | **優化分割演算法**：發現單純的 K-means 分割效果不佳，改進為 K-means 結合輪廓面積尋找，顯著提升了分割準確度。 |
| **改進 4** | PCA + GMM + 優化分割 + **規則限制** | **71.43%** | **導入領域知識**：利用台灣車牌「前3英文字母、後4數字」的固定規則，過濾不合理的預測結果。 |
| **最終循環** | (同上) + **基於頻率統計優化資料集** | **78.10%** | **精煉訓練資料**：對辨識結果進行字元頻率統計，修正基礎資料集並重新訓練模型，達到最終辨識率。 |

## 資料集

本專案主要使用了以下資料來源：

1.  **Kaggle - License Plate Digits Classification Dataset**：專案初期用於訓練線性分類器的資料集。
2.  **roboflow - High Precision Taiwan License Plate Number Recognition**：用於最終模型評估的測試資料集。
3.  **自建台灣車牌字元資料集**：
      - **來源**：參考[台灣車牌規則](https://cynthiachuang.github.io/Taiwan-License-Plate-Rules-for-LPR/)，手動裁切建立 (我們選擇第八代的字元樣式進行訓練)。
      - **處理**：對每個字元進行縮放、旋轉和添加噪音等**資料增強**，將資料量擴增 1000 倍，以提高模型的泛化能力。
      - **優化**：在專案後期，根據字元辨識的頻率統計結果對此資料集進行了多次修正。

## 自建訓練資料集流程

本專案提供了完整的自建訓練資料集流程，從原始圖片到最終模型訓練的完整步驟：

### 步驟 1：從參考圖裁切字元
使用 `img_cropper.py` 從 `TW_plate_style.jpg` 手動裁切各個字元：

```bash
python img_cropper.py
```
- 開啟互動式圖片裁剪工具
- 手動選取每個字元區域（0-9, A-Z，共35個字元）
- 自動儲存到 `TW_plate_digits/` 資料夾

### 步驟 2：建立原始資料集結構
使用 `createDir.py` 整理資料夾結構：

```bash
python createDir.py
```
- 將裁切好的字元圖片按檔名分類到對應子資料夾
- 建立 `TW_plate_digits/` 下的字元資料夾結構
- 參考 `hyse/TW_plate_digits (origin)/` 的資料夾組織方式

### 步驟 3：資料增強
使用 `data_augmentation.py` 進行資料擴增：

```bash
python data_augmentation.py
```
- **輸入路徑**：`TW_plate_digits/`（原始字元資料夾）
- **輸出路徑**：同一個資料夾（直接覆蓋擴增）
- **增強方法**：
  - 隨機縮放（0.95-1.00）
  - 隨機旋轉（-6°到+6°）
  - 高斯噪音（20%機率）
- **擴增倍數**：每個字元生成1000張增強圖片

### 步驟 4：模型訓練
使用 `hyse/train_model.py` 訓練 PCA + GMM 模型：

```bash
cd hyse
python train_model.py
```
- **資料集路徑**：`D:\license_plate_datasets\TW_plate_digits`（需修改為實際路徑）
- **輸出模型**：
  - `models/tw_pca_model_demo.pkl`（PCA模型）
  - `models/tw_gmm_models_demo/`（各字元的GMM模型）
- **模型參數**：
  - PCA維度：50
  - GMM組件數：3
  - 字元尺寸：36×18

### 完整流程指令

```bash
# 1. 裁切字元（手動操作）
python img_cropper.py

# 2. 整理資料夾結構
python createDir.py

# 3. 資料增強
python data_augmentation.py

# 4. 訓練模型
cd hyse
python train_model.py
```

完成後即可獲得訓練好的 PCA 和 GMM 模型，用於後續的車牌識別任務。

## 檔案結構與說明

```
.
├── Roy/                          # 方法一：線性分類器
│   ├── ML_final_proj.py          # 主要識別程式 (K-means + 線性分類)
│   ├── inference.py              # 字元識別推論模組
│   ├── classification_training.py# 線性分類器訓練程式
│   ├── weights.npy               # 訓練好的權重
│   └── biases.npy                # 訓練好的偏置
│
├── hyse/                         # 方法二：PCA + GMM (主要開發目錄)
│   ├── train_model.py            # PCA + GMM 模型訓練程式
│   ├── recognize_plate*.py       # 多個版本的車牌識別程式
│   ├── utils*.py                 # 不同版本的工具函數
│   ├── models/                   # 存放訓練好的 PCA 與 GMM 模型
│   ├── img/                      # 測試圖片
│   ├── test_results/             # 測試結果紀錄
│   └── TW_plate_digits/          # 字元資料集
│
├── TW_plate_digits/              # 原始字元資料集
├── data_augmentation.py          # 資料增強工具
├── img_cropper.py                # 互動式圖片裁剪工具
├── createDir.py                  # 資料夾管理工具
├── toGray.py                     # 圖片轉灰階與二值化工具
├── TW_plate_style.jpg            # 台灣車牌樣式參考圖
└── README.md                     # 本說明文件
```

### `hyse` 資料夾內主要程式演進

`hyse` 資料夾記錄了從基礎到最終版本的完整開發歷程，不同檔案代表了不同階段的嘗試與優化。

#### 版本演進對比

| 版本 (`recognize_plate_*.py`) | 車牌定位方法 | 字元分割策略 | 規則限制 | 主要改進 |
| :--- | :--- | :--- | :---: | :--- |
| **origin** | Canny + 輪廓 | K-means + 輪廓 | ❌ | 基礎版本 |
| **GMM** | Canny + 輪廓 | 純 K-means | ❌ | 為 GMM 簡化分割邏輯 |
| **contour** | Canny + 輪廓 | K-means + 輪廓 | ❌ | 專注於輪廓分割優化 |
| **v2** | **Sobel + 垂直投影** | K-means + 輪廓 | ✅ | **定位方法改進**，更穩定 |
| **recognize\_plate.py (最新)** | Canny + 輪廓 | K-means + 輪廓 | ✅ | **功能最完整**，包含規則限制與詳細評估 |

#### 關鍵差異總結

1.  **車牌定位方法**：從早期的 `Canny + 輪廓` 演進到後期的 `Sobel + 垂直投影`，後者在定位上更為穩定。
2.  **字元分割策略**：從純 `K-means` 演進到 `K-means + 輪廓分析`，後者分割更精確。
3.  **台灣車牌規則**：在後期版本中加入「前3字母後4數字」的限制，是辨識率大幅提升的關鍵。
