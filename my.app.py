import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from umap import UMAP
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests

# 页面配置
st.set_page_config(page_title="Gold mineralization prediction", layout="wide")
st.title("Gold Mineralization Prediction")

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Times New Roman", Times, serif !important;
}

.title-text {
    font-size: 26px;
    font-weight: 700;
    color: #333333;
    margin-bottom: 10px;
}

.dev-text {
    font-size: 18px;
    font-weight: 500;
    color: #000000;
    line-height: 2;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-text">Prediction of Gold Mineralization Potential in Magmatic Rocks</div>', unsafe_allow_html=True)
st.markdown('<div class="dev-text">•   Developer:  Dr. Fengge Han;   School of Science, East China University of Science and Technology, Nanchang 330013, China</div>', unsafe_allow_html=True)
st.markdown('<div class="dev-text">•   Developer:  Prof. Chengbiao Leng;   School of Earth and Planetary Sciences, East China University of Science and Technology, Nanchang 330013, China</div>', unsafe_allow_html=True)
st.markdown('<div class="dev-text">•   Developer:  Assoc. Prof. Jiajie Chen;   School of Earth and Planetary Sciences, East China University of Science and Technology, Nanchang 330013, China</div>', unsafe_allow_html=True)
st.write("##### •   Email: hanfengge@ecut.edu.cn(Han F.G.)")

# 绿色波浪线
st.markdown(
    """<hr style="border: 0; border-top: 2px solid green; width: 100%; background-image: url('https://upload.wikimedia.org/wikipedia/commons/a/a5/Wave_pattern.svg'); height: 10px;">""",
    unsafe_allow_html=True
)

# 模型加载提示
st.subheader("Firstly:⚙️Model loading, please wait......")

# ===== 特征列和标签列 =====
feature_columns = [
    "SiO2", "TiO2", "Al2O3", "FeOt", "MnO", "MgO",
    "CaO", "Na2O", "K2O", "P2O5", "Rb", "Ba", "Nb",
    "Sr", "Zr", "Ba/Zr", "Nb/Zr"
]
target_column = "Label"  # CSV 中是 "Label" 大写

# ===== 训练模型（只训练一次） =====
@st.cache_data(show_spinner=True)
def train_model():
    train_file = "https://raw.githubusercontent.com/fenggeHan/gold-prediction-app/refs/heads/main/xunlian1754.csv"
    data = pd.read_csv(train_file)
    label_map = {"Fertile": 1, "Barren": 0}
    data[target_column] = data[target_column].map(label_map)

    X = data[feature_columns].values
    y = data[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric='euclidean', random_state=42)
    X_train_umap = umap_model.fit_transform(X_train_scaled)
    X_test_umap = umap_model.transform(X_test_scaled)

    wrf_model = RandomForestClassifier(
        max_features='sqrt',
        max_depth=None,
        min_samples_leaf=2,
        min_samples_split=2,
        class_weight='balanced',
        random_state=42
    )
    wrf_model.fit(X_train_umap, y_train)

    train_predictions = wrf_model.predict(X_train_umap)
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_predictions = wrf_model.predict(X_test_umap)
    test_accuracy = accuracy_score(y_test, test_predictions)

    return scaler, umap_model, wrf_model, train_accuracy, test_accuracy

scaler, umap_model, wrf_model, train_accuracy, test_accuracy = train_model()
st.success("Model training completed")

# 绿色波浪线
st.markdown(
    """<hr style="border: 0; border-top: 2px solid green; width: 100%; background-image: url('https://upload.wikimedia.org/wikipedia/commons/a/a5/Wave_pattern.svg'); height: 10px;">""",
    unsafe_allow_html=True
)

# ===== 下载模板 =====
st.subheader("Secondly: 📥 Download Data Template")
template_url = "https://raw.githubusercontent.com/fenggeHan/gold-prediction-app/refs/heads/main/Data%20Template.csv"
response = requests.get(template_url)
if response.status_code == 200:
    st.download_button(
        label="Download Data Template (CSV)",
        data=response.content,
        file_name="Data_Template.csv",
        mime="text/csv"
    )
else:
    st.error("❌ 模板文件无法从 GitHub 加载，请检查文件是否存在。")
st.info("Template download completed!")
# 蓝色波浪线
st.markdown(
    """<hr style="border: 0; border-top: 2px solid blue; width: 100%; background-image: url('https://upload.wikimedia.org/wikipedia/commons/a/a5/Wave_pattern.svg'); height: 10px;">""",
    unsafe_allow_html=True
)

# ===== 上传新数据预测 =====
st.markdown("### Thirdly: 📁 Upload new data CSV (17 features) for prediction, please download the data template!")
new_file = st.file_uploader("Please upload a CSV file that matches the template", type=["csv"])

if new_file is not None:
    new_data = pd.read_csv(new_file)
    st.write("📊 New Data Preview：", new_data.head())

    missing_cols = [col for col in feature_columns if col not in new_data.columns]
    if missing_cols:
        st.error(f"The uploaded new data lacks necessary feature columns：{missing_cols}")
    else:
        X_new = new_data[feature_columns].values
        X_new_scaled = scaler.transform(X_new)
        X_new_umap = umap_model.transform(X_new_scaled)

        # 预测及置信水平
        predictions_raw = wrf_model.predict(X_new_umap)
        predictions_proba = wrf_model.predict_proba(X_new_umap)
        inv_label_map = {1: "Fertile", 0: "Barren"}

        predictions = [inv_label_map[p] for p in predictions_raw]
        confidence = [round(probas[pred_idx]*100, 2) for probas, pred_idx in zip(predictions_proba, predictions_raw)]

        new_data["Prediction"] = predictions
        new_data["Confidence (%)"] = confidence
        st.success("📈 Prediction completed！")

        def highlight_prediction(val):
            if val == "Fertile":
                color = "red"
            else:
                color = "green"
            return f"color: {color}; font-weight: bold"

        styled_df = new_data.style.applymap(highlight_prediction, subset=["Prediction"])
        st.dataframe(styled_df, use_container_width=True)

        st.markdown("### Finally:📥💾 Download prediction results")
        output_csv = new_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=" Download prediction results",
            data=output_csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )
else:
    st.warning("Please check if your file format is correct and upload a CSV file that matches the model.")

# 黄色虚线
st.markdown("""<hr style="border: 0; border-top: 2px dashed yellow; width: 100%; height: 1px;">""", unsafe_allow_html=True)

st.subheader("Citation and Funding")
st.write("###### * Han, F., Leng, C., & Chen, J.(contributor). Machine Learning-Driven Mineral Prospectivity Modeling for Intrusion-Related Gold Deposits（Under review）")
st.write("###### * This work was co-funded by the National Science and Technology Major Project (Grant No. 2025ZD1009303) and the National Science and Technology Major Project (Grant No. 2024ZD1001602) .")

st.markdown("""
<div style="text-align: center; padding: 20px; font-size: 24px; color: #4CAF50;">
    🌟🌟🌟 *** Thank you for using our service! May your research yield great results and lead to a bright future! *** 🌟🌟🌟
</div>
""", unsafe_allow_html=True)

# 绿色波浪线
st.markdown("""<hr style="border: 0; border-top: 2px solid green; width: 100%; background-image: url('https://upload.wikimedia.org/wikipedia/commons/a/a5/Wave_pattern.svg'); height: 10px;">""", unsafe_allow_html=True)
