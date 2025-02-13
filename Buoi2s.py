import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 📌 Cấu hình MLflow Tracking URI (Sử dụng URI từ xa nếu có)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://your-mlflow-server")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 📂 Đọc dữ liệu gốc từ URL hoặc tải lên
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"  # Thay URL dữ liệu phù hợp
DATA_PATH = "data.csv"

if not os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_URL)
    df.to_csv(DATA_PATH, index=False)  # Lưu để dùng sau
else:
    df = pd.read_csv(DATA_PATH)

# 🔹 Xử lý dữ liệu bị thiếu (NaN)
df['Age'] = df['Age'].fillna(df['Age'].median())
df.dropna(subset=['Embarked'], inplace=True)
df['Cabin'] = df['Cabin'].fillna('Unknown')

# 🔹 Chuyển đổi dữ liệu dạng chuỗi thành số
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# 🔹 Loại bỏ các cột không cần thiết
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# 🔹 Xử lý Outliers (ngoại lệ)
Q1, Q3 = df['Fare'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

# 🔹 Chuẩn hóa dữ liệu
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 📌 Lưu dữ liệu đã xử lý
PROCESSED_PATH = "processed_data.csv"
df.to_csv(PROCESSED_PATH, index=False)

# 🔹 Chia tập dữ liệu
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

# 📌 Ghi logs vào MLflow
mlflow.set_experiment("Titanic_Data_Preprocessing")
if mlflow.active_run():
    mlflow.end_run()

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
    mlflow.log_artifact(PROCESSED_PATH)
    mlflow.end_run()

# 📌 Hiển thị Dashboard trên Streamlit
st.title("🚢 Titanic Data Preprocessing Dashboard")

# 📂 Đọc dữ liệu đã xử lý
df = pd.read_csv(PROCESSED_PATH)

# 📌 Lấy run ID gần nhất từ MLflow
experiment = mlflow.get_experiment_by_name("Titanic_Data_Preprocessing")
if experiment:
    experiment_id = experiment.experiment_id
    latest_run = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1)
    if latest_run.empty:
        st.warning("⚠️ Không tìm thấy thông tin từ MLflow!")
        run_data = None
    else:
        run_id = latest_run.iloc[0]["run_id"]
        run_data = mlflow.get_run(run_id).data
else:
    run_data = None

# 📌 Lấy thông số từ MLflow
if run_data:
    train_size = int(run_data.metrics.get("train_size", 0))
    val_size = int(run_data.metrics.get("val_size", 0))
    test_size = int(run_data.metrics.get("test_size", 0))
    mlflow_params = run_data.params
else:
    train_size, val_size, test_size = 0, 0, 0
    mlflow_params = {}

# 📊 Hiển thị DataFrame sau xử lý
st.subheader("🔹 Dữ liệu sau khi tiền xử lý")
st.dataframe(df.head())

# 📌 Hiển thị thông tin tập dữ liệu
st.subheader("📊 Thông tin kích thước tập dữ liệu")
st.write(f"**🔹 Training size:** {train_size}")
st.write(f"**🔸 Validation size:** {val_size}")
st.write(f"**🔹 Test size:** {test_size}")

# 📊 Vẽ biểu đồ tỷ lệ tập dữ liệu
if train_size > 0 and val_size > 0 and test_size > 0:
    sizes = [train_size, val_size, test_size]
    labels = ["Train", "Validation", "Test"]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=['#3498db', '#f39c12', '#2ecc71'], startangle=90)
    ax.set_title("📊 Tỉ lệ tập dữ liệu")
    st.pyplot(fig)
else:
    st.warning("⚠️ Không thể hiển thị biểu đồ.")

# 🛠 Hiển thị thông số MLflow
st.subheader("📜 Thông tin từ MLflow")
if mlflow_params:
    st.json(mlflow_params)
else:
    st.warning("⚠️ Không có thông số nào từ MLflow.")

# ✅ Kết thúc ứng dụng
st.success("🎉 Dữ liệu đã được hiển thị thành công!")
