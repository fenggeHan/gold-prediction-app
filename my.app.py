import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from umap import UMAP

# 页面配置
st.set_page_config(page_title="Gold mineralization prediction", layout="wide")
st.title("Gold Mineralization Prediction")
st.write("Using UMAP+WRF model to perform gold Fertile/Barren prediction on newly uploaded whole rock data of magmatic rocks")

# ===== 特征列和标签列 =====
feature_columns = [
    "SiO2", "TiO2", "Al2O3", "FeOt", "MnO", "MgO",
    "CaO", "Na2O", "K2O", "P2O5", "Rb", "Ba", "Nb",
    "Sr", "Zr", "Ba/Zr", "Nb/Zr"
]
target_column = "Label"  # 注意 CSV 中是 "Label" 大写

# ===== 训练模型（只训练一次） =====
@st.cache_data(show_spinner=True)
def train_model():
    train_file = "https://raw.githubusercontent.com/fenggeHan/gold-prediction-app/refs/heads/main/xunlian1754.csv"

    # 读取 CSV
    data = pd.read_csv(train_file)

    # 标签映射为 0/1
    label_map = {"Fertile": 1, "Barren": 0}
    data[target_column] = data[target_column].map(label_map)

    X_train = data[feature_columns].values
    y_train = data[target_column].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # UMAP降维
    umap_model = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric='euclidean',
        random_state=42
    )
    X_umap = umap_model.fit_transform(X_scaled)

    # WRF训练
    wrf_model = RandomForestClassifier(
        max_features='sqrt',
        max_depth=None,
        min_samples_leaf=2,
        min_samples_split=2,
        class_weight='balanced',
        random_state=42
    )
    wrf_model.fit(X_umap, y_train)

    return scaler, umap_model, wrf_model

# 训练模型
scaler, umap_model, wrf_model = train_model()
st.success("Model training completed")

# ===== 上传新数据进行预测 =====
new_file = st.file_uploader("Upload new data CSV (17 features) for prediction, please download the data template!", type=["csv"])

if new_file is not None:
    new_data = pd.read_csv(new_file)
    st.write("New Data Preview：", new_data.head())

    # 检查列完整性
    missing_cols = [col for col in feature_columns if col not in new_data.columns]
    if missing_cols:
        st.error(f"上传的新数据缺少必要特征列：{missing_cols}")
    else:
        X_new = new_data[feature_columns].values

        # 标准化
        X_new_scaled = scaler.transform(X_new)

        # UMAP降维
        X_new_umap = umap_model.transform(X_new_scaled)

        # 预测
        predictions_raw = wrf_model.predict(X_new_umap)
        inv_label_map = {1: "Fertile", 0: "Barren"}
        predictions = [inv_label_map[p] for p in predictions_raw]

        new_data["Prediction"] = predictions
        st.success("Prediction completed！")

        # 彩色表格显示
        def highlight_prediction(val):
            if val == "Fertile":
                color = "red"
            else:
                color = "green"
            return f"color: {color}; font-weight: bold"

        styled_df = new_data.style.applymap(highlight_prediction, subset=["Prediction"])
        st.dataframe(styled_df, use_container_width=True)

        # 下载结果
        output_csv = "prediction_results.csv"
        new_data.to_csv(output_csv, index=False)
        with open(output_csv, "rb") as f:
            st.download_button(
                label="Download prediction results",
                data=f,
                file_name="prediction_results.csv",
                mime="text/csv"
            )


