import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 📌 Cấu hình MLflow Tracking URI (chạy local)
mlflow.set_tracking_uri("file:///C:/TraThanhTri/PYthon/TriTraThanh/MLvsPython/mlruns")

# 📂 **Đọc dữ liệu**
df = pd.read_csv("C:/TraThanhTri/PYthon/TriTraThanh/MLvsPython/data.csv", encoding="utf-8")

### 1️⃣ Kiểm tra thông tin dữ liệu
print(df.info())
print(df.head())

# 🔹 Kiểm tra số lượng giá trị thiếu
print(df.isnull().sum())

### 2️⃣ Xử lý dữ liệu bị thiếu (NaN)
df['Age'] = df['Age'].fillna(df['Age'].median())
df.dropna(subset=['Embarked'], inplace=True)
df['Cabin'] = df['Cabin'].fillna('Unknown')

# 🔹 Chuyển đổi dữ liệu dạng chuỗi thành số
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# 🔹 Loại bỏ các cột không cần thiết
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

### 3️⃣ Xử lý Outliers (ngoại lệ)
Q1, Q3 = df['Fare'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

# 🔹 **Chuẩn hóa dữ liệu**
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

### 4️⃣ Kết quả sau tiền xử lý
print(df.head())
print(df.info())

# 📌 **Lưu dữ liệu sau tiền xử lý**
processed_file = "C:/TraThanhTri/PYthon/processed_data.csv"
df.to_csv(processed_file, index=False)

### 5️⃣ Chia tập dữ liệu
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

# 📌 **Kiểm tra kích thước tập dữ liệu**
print(f"Training size: {X_train.shape[0]}")
print(f"Validation size: {X_val.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# 🛠 **Ghi logs vào MLflow**
mlflow.set_experiment("Titanic_Data_Preprocessing")

if mlflow.active_run():
    mlflow.end_run()  # Kết thúc run cũ nếu có

with mlflow.start_run():
    mlflow.log_param("fillna_age", "median")
    mlflow.log_param("dropna_embarked", True)
    mlflow.log_param("fillna_cabin", "Unknown")
    mlflow.log_param("sex_encoding", "male=1, female=0")
    mlflow.log_param("embarked_encoding", "S=0, C=1, Q=2")
    mlflow.log_param("drop_columns", "Name, Ticket, Cabin")
    mlflow.log_param("outlier_detection", "IQR for Fare")
    mlflow.log_param("scaling_method", "StandardScaler")
    mlflow.log_metric("train_size", X_train.shape[0])
    mlflow.log_metric("val_size", X_val.shape[0])
    mlflow.log_metric("test_size", X_test.shape[0])

    # 📂 Ghi file vào MLflow
    mlflow.log_artifact(processed_file)

    # ✅ Kết thúc run
    mlflow.end_run()
