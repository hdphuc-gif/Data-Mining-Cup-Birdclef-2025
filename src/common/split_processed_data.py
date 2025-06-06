# src/common/split_processed_data.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Đường dẫn dữ liệu
PROCESSED_DIR = r'D:\Studi\Projects\data\processed'
TRAIN_CSV = r"D:\Studi\Projects\data\train_a.csv"
VAL_CSV = r"D:\Studi\Projects\data\val_a.csv"

# Tạo danh sách tất cả file .wav và label
records = []
for label in os.listdir(PROCESSED_DIR):
    label_dir = os.path.join(PROCESSED_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    for fname in os.listdir(label_dir):
        if fname.endswith(".wav"):
            filepath = os.path.join(label_dir, fname)
            records.append({
                "filepath": filepath.replace("\\", "/"),  # để tương thích đa nền tảng
                "label": label
            })

# Chuyển thành DataFrame
df = pd.DataFrame(records)
print(f"📦 Tổng số file: {len(df)}")
print(f"📊 Số class: {df['label'].nunique()}")

# Stratified split
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# Lưu ra CSV
os.makedirs("data", exist_ok=True)
train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
print(f"✅ Đã lưu: {TRAIN_CSV}, {VAL_CSV}")
