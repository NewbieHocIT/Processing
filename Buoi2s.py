import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 📌 Đọc dữ liệu từ URL
DATA_URL = "data.csv"
df = pd.read_csv(DATA_URL)

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

# 🔹 Chia tập dữ liệu
X = df.drop(columns=['Survived'])
y = df['Survived']
# 📌 Chia tập dữ liệu thành 70% Train và 30% Test + Validation
X_train, X_test_val, y_train, y_test_val = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 📌 Chia tiếp tập Test + Validation thành 15% Test và 15% Validation
X_test, X_val, y_test, y_val = train_test_split(
    X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val
)



# 📌 Hiển thị Dashboard trên Streamlit
st.title("🚢 Titanic Data Preprocessing Dashboard")
# 📌 Hiển thị chi tiết các bước xử lý dữ liệu
st.subheader("📌 Các bước xử lý dữ liệu")

with st.expander("🔹 1. Xử lý dữ liệu bị thiếu (NaN)"):
    st.markdown("""
    - **Cột `Age`**: Điền giá trị thiếu bằng **median (trung vị)**.
    - **Cột `Embarked`**: Loại bỏ các dòng bị thiếu dữ liệu.
    - **Cột `Cabin`**: Điền giá trị thiếu bằng `"Unknown"`.
    """)

with st.expander("🔹 2. Chuyển đổi dữ liệu dạng chuỗi thành số"):
    st.markdown("""
    - **Cột `Sex`**: Chuyển thành số (`male=1, female=0`).
    - **Cột `Embarked`**: Chuyển thành số (`S=0, C=1, Q=2`).
    """)

with st.expander("🔹 3. Loại bỏ cột không cần thiết"):
    st.markdown("""
    - Loại bỏ các cột: **`Name`**, **`Ticket`**, **`Cabin`** (không ảnh hưởng đến dự đoán).
    """)

with st.expander("🔹 4. Xử lý ngoại lệ trong giá vé (`Fare`)"):
    st.markdown("""
    - Áp dụng phương pháp **IQR (Interquartile Range)** để loại bỏ các giá trị ngoại lệ.
    """)

with st.expander("🔹 5. Chuẩn hóa dữ liệu"):
    st.markdown("""
    - Áp dụng **StandardScaler** để chuẩn hóa **`Age`** và **`Fare`** về phân phối chuẩn.
    """)

with st.expander("🔹 6. Chia tập dữ liệu"):
    st.markdown("""
    - **70% dữ liệu** dùng để huấn luyện (`Train`).
    - **15% dữ liệu** dùng để kiểm định (`Validation`).
    - **15% dữ liệu** dùng để kiểm tra (`Test`).
    """)

st.success("📌 Các bước xử lý dữ liệu đã được hiển thị chi tiết!") 

# 📊 Hiển thị DataFrame sau xử lý
st.subheader("🔹 Dữ liệu sau khi tiền xử lý")
st.dataframe(df.head())

# 📌 Hiển thị thông tin tập dữ liệu
st.subheader("📊 Thông tin kích thước tập dữ liệu")
st.write(f"**🔹 Training size:** {X_train.shape[0]}")
st.write(f"**🔸 Validation size:** {X_val.shape[0]}")
st.write(f"**🔹 Test size:** {X_test.shape[0]}")

# 📊 Vẽ biểu đồ tỷ lệ tập dữ liệu
sizes = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
labels = ["Train", "Validation", "Test"]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=['#3498db', '#f39c12', '#2ecc71'], startangle=90)
ax.set_title("📊 Tỉ lệ tập dữ liệu")
st.pyplot(fig)

# ✅ Kết thúc ứng dụng
st.success("🎉 Dữ liệu đã được xử lý và hiển thị thành công!")
