#### 將資料夾中的圖片依檔名建立子資料夾並移動圖片到對應的子資料夾中
# import os
# import shutil

# # 設定資料夾路徑
# folder_path = 'TW_plate_digits'

# # 取得所有 .jpg 檔案
# for filename in os.listdir(folder_path):
#     if filename.lower().endswith('.jpg'):
#         name = os.path.splitext(filename)[0]
#         img_path = os.path.join(folder_path, filename)
#         new_dir = os.path.join(folder_path, name)
#         # 建立新資料夾
#         os.makedirs(new_dir, exist_ok=True)
#         # 移動檔案
#         shutil.move(img_path, os.path.join(new_dir, filename))

#### 將子資料夾中所有 augmentation 的檔案刪除
import os

base_dir = '../TW_plate_digits'

for subfolder in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder)
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            if 'aug' in filename:
                file_path = os.path.join(subfolder_path, filename)
                try:
                    os.remove(file_path)
                    print(f"已刪除：{file_path}")
                except Exception as e:
                    print(f"刪除失敗：{file_path}，原因：{e}")