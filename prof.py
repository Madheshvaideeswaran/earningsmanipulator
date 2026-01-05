import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Earnings Manipulation Detector",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# HIGH-CONTRAST YELLOW THEME (FIXED VISIBILITY)
# ---------------------------------------------------
st.markdown("""
<style>

/* GLOBAL RESET */
html, body, [class*="css"]  {
    color: #111827 !important;
}

/* ENTIRE APP BACKGROUND */
.stApp {
    background-color: #fff4cc;
    color: #111827;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #ffe082;
    border-right: 1px solid #d4b100;
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: #111827 !important;
    font-weight: 500;
}

/* HEADINGS */
h1 {
    color: #0f172a !important;
    font-weight: 800;
}
h2, h3, h4 {
    color: #111827 !important;
    font-weight: 700;
}

/* PARAGRAPH / MARKDOWN TEXT */
p, li, span, label, small {
    color: #111827 !important;
    font-weight: 500;
}

/* INPUT LABELS */
label {
    font-weight: 600 !important;
}

/* SELECTBOX / SLIDER TEXT */
.stSelectbox *, .stSlider * {
    color: #111827 !important;
}

/* BUTTONS */
.stButton > button {
    background-color: #1d4ed8;
    color: #ffffff !important;
    border-radius: 6px;
    border: none;
    padding: 0.55rem 1.3rem;
    font-weight: 700;
}
.stButton > button:hover {
    background-color: #1e40af;
}

/* METRIC CARDS */
[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    padding: 14px;
    box-shadow: 0 3px 6px rgba(0,0,0,0.06);
}

/* METRIC TEXT */
[data-testid="stMetricLabel"] {
    color: #1f2937 !important;
    font-weight: 700;
}
[data-testid="stMetricValue"] {
    color: #020617 !important;
    font-weight: 800;
}

/* FILE UPLOADER */
section[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border: 2px dashed #ca8a04;
    border-radius: 10px;
    padding: 16px;
}
section[data-testid="stFileUploader"] * {
    color: #111827 !important;
}

/* DATAFRAME */
.stDataFrame, .stTable {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 10px;
}
.stDataFrame * {
    color: #111827 !important;
}

/* ALERTS */
.stAlert * {
    color: #111827 !important;
    font-weight: 600;
}

/* CODE BLOCK */
code {
    color: #020617 !important;
    background-color: #fef3c7 !important;
    border-radius: 6px;
    padding: 4px 6px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------------
def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

# ---------------------------------------------------
# PARAMETER GRIDS
# ---------------------------------------------------
PARAM_GRIDS = {
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
    "KNN": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
    "Naive Bayes": {"var_smoothing": np.logspace(0, -9, 10)},
    "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5]},
    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 4, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
}

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def needs_scaling(model_name):
    return model_name in ["SVM", "KNN"]

def get_model(model_name):
    if model_name == "SVM":
        return SVC(probability=True, random_state=42)
    elif model_name == "KNN":
        return KNeighborsClassifier()
    elif model_name == "Naive Bayes":
        return GaussianNB()
    elif model_name == "AdaBoost":
        return AdaBoostClassifier(random_state=42)
    elif model_name == "XGBoost":
        return XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False
        )

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("Earnings Manipulation Detection Dashboard")

st.markdown("""
This dashboard applies **machine learning classification models**  
to identify potential **earnings manipulation** using Beneish indicators.

**Required columns:**  
`DSRI`, `GMI`, `AQI`, `SGI`, `DEPI`, `SGAI`, `ACCR`, `LEVI`, `Manipulator (Yes / No)`
""")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    required_cols = ["DSRI","GMI","AQI","SGI","DEPI","SGAI","ACCR","LEVI","Manipulator"]
    if not all(col in df.columns for col in required_cols):
        st.error("Missing required columns.")
        st.stop()

    X = df[required_cols[:-1]]
    y = df["Manipulator"].map({"No": 0, "Yes": 1})

    st.sidebar.header("Model Configuration")
    model_name = st.sidebar.selectbox(
        "Select Algorithm",
        ["SVM", "KNN", "Naive Bayes", "AdaBoost", "XGBoost"]
    )
    test_size = st.sidebar.slider("Test Set Proportion", 0.1, 0.5, 0.25, step=0.05)
    use_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning")
    run = st.sidebar.button("Run Model")

    if run:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        if needs_scaling(model_name):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = get_model(model_name)

        if use_tuning:
            grid = GridSearchCV(
                model,
                PARAM_GRIDS[model_name],
                cv=5,
                scoring="roc_auc",
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_

        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_prob)

        st.subheader("Model Performance")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            st.metric("Precision", f"{metrics['Precision']:.4f}")
            st.metric("Recall", f"{metrics['Recall']:.4f}")

        with col2:
            st.metric("F1-score", f"{metrics['F1-score']:.4f}")
            st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")

        st.subheader("Beneish M-Score Baseline")
        st.dataframe(pd.DataFrame([{
            "Accuracy": 0.8364,
            "Precision": 0.5556,
            "Recall": 0.5000,
            "F1-score": 0.5263,
            "ROC-AUC": 0.9044
        }]).round(4))

else:
    st.info("Upload an Excel file to begin analysis.")
