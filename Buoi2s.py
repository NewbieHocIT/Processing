import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os

# 📌 **Cấu hình MLflow Tracking URI**
mlflow_dir = "./mlruns"  # Lưu dữ liệu MLflow trong thư mục hiện tại
os.makedirs(mlflow_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_dir)}")

# 📂 **Đọc dữ liệu đã xử lý**
processed_file = "processed_data.csv"
if os.path.exists(processed_file):
    df = pd.read_csv(processed_file)
else:
    st.error("🚨 Không tìm thấy file dữ liệu đã xử lý! Hãy chắc chắn rằng quá trình tiền xử lý đã được chạy.")
    st.stop()

# 📌 **Lấy hoặc tạo `experiment_id`**
experiment = mlflow.get_experiment_by_name("Titanic_Data_Preprocessing")
if experiment:
    experiment_id = experiment.experiment_id
else:
    experiment_id = mlflow.create_experiment("Titanic_Data_Preprocessing")

# 📌 **Lấy run ID gần nhất từ MLflow**
latest_run = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=1)

if latest_run.empty:
    st.warning("⚠️ Không tìm thấy thông tin từ MLflow! Một số dữ liệu có thể không hiển thị đầy đủ.")
    run_data = None
else:
    run_id = latest_run.iloc[0]["run_id"]
    run_data = mlflow.get_run(run_id).data

# 📌 **Lấy thông số từ MLflow**
if run_data:
    train_size = int(run_data.metrics.get("train_size", 0))
    val_size = int(run_data.metrics.get("val_size", 0))
    test_size = int(run_data.metrics.get("test_size", 0))
    mlflow_params = run_data.params
else:
    train_size, val_size, test_size = 0, 0, 0
    mlflow_params = {}

# 🏷️ **Tiêu đề ứng dụng**
st.title("🚢 Titanic Data Preprocessing Dashboard")

# 📊 **Hiển thị DataFrame sau xử lý**
st.subheader("🔹 Dữ liệu sau khi tiền xử lý")
st.dataframe(df.head())

# 📌 **Hiển thị thông tin tập dữ liệu**
st.subheader("📊 Thông tin kích thước tập dữ liệu")
st.write(f"**🔹 Training size:** {train_size}")
st.write(f"**🔸 Validation size:** {val_size}")
st.write(f"**🔹 Test size:** {test_size}")

# 📊 **Vẽ biểu đồ tỷ lệ tập dữ liệu**
if train_size > 0 and val_size > 0 and test_size > 0:
    sizes = [train_size, val_size, test_size]
    labels = ["Train", "Validation", "Test"]
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=['#3498db', '#f39c12', '#2ecc71'], startangle=90)
    ax.set_title("📊 Tỉ lệ tập dữ liệu")
    st.pyplot(fig)
else:
    st.warning("⚠️ Không thể hiển thị biểu đồ do thiếu thông tin về kích thước dữ liệu.")

# 🛠️ **Hiển thị thông số MLflow**
st.subheader("📜 Thông tin từ MLflow")
if mlflow_params:
    st.json(mlflow_params)
else:
    st.warning("⚠️ Không có thông số nào được lưu trong MLflow.")

# ✅ **Kết thúc ứng dụng**
st.success("🎉 Dữ liệu đã được hiển thị thành công!")
