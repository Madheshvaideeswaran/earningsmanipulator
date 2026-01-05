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
# CUSTOM YELLOW-BLACK THEME (CSS)
# ---------------------------------------------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #0f0f0f;
    color: #f5c518;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1a1a1a;
}

/* Headers */
h1, h2, h3, h4 {
    color: #f5c518;
}

/* Buttons */
.stButton > button {
    background-color: #f5c518;
    color: black;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #ffd84d;
    color: black;
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: #1f1f1f;
    border-left: 5px solid #f5c518;
    padding: 10px;
    border-radius: 8px;
}

/* Dataframe */
.stDataFrame {
    background-color: #1f1f1f;
}

/* File uploader */
section[data-testid="stFileUploader"] {
    background-color: #1f1f1f;
    border-radius: 10px;
    padding: 10px;
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
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"]
    },
    "Naive Bayes": {
        "var_smoothing": np.logspace(0, -9, 10)
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5]
    },
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
st.title("📊 Earnings Manipulation Classification Dashboard")
st.markdown("""
Upload your **financial dataset** to detect earnings manipulation using machine learning models.

**Required columns:**  
`DSRI`, `GMI`, `AQI`, `SGI`, `DEPI`, `SGAI`, `ACCR`, `LEVI`, `Manipulator (Yes/No)`
""")

uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        required_cols = [
            "DSRI", "GMI", "AQI", "SGI", "DEPI",
            "SGAI", "ACCR", "LEVI", "Manipulator"
        ]

        if not all(col in df.columns for col in required_cols):
            st.error("❌ Missing required columns in uploaded file.")
            st.stop()

        X = df[required_cols[:-1]]
        y = df["Manipulator"].map({"No": 0, "Yes": 1})

        if y.isnull().any():
            st.error("❌ 'Manipulator' must contain only 'Yes' or 'No'")
            st.stop()

        # Sidebar
        st.sidebar.header("⚙️ Model Settings")
        model_name = st.sidebar.selectbox(
            "Select Model",
            ["SVM", "KNN", "Naive Bayes", "AdaBoost", "XGBoost"]
        )
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.25, step=0.05)
        use_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning")
        run = st.sidebar.button("🚀 Run Model")

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
                with st.spinner("🔍 Tuning model..."):
                    grid = GridSearchCV(
                        model,
                        PARAM_GRIDS[model_name],
                        cv=5,
                        scoring="roc_auc",
                        n_jobs=-1
                    )
                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_
                    best_params = grid.best_params_
            else:
                model.fit(X_train, y_train)
                best_params = "Default parameters"

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = evaluate(y_test, y_pred, y_prob)

            st.subheader(f"📈 Model Performance – {model_name}")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                st.metric("Precision", f"{metrics['Precision']:.4f}")
                st.metric("Recall", f"{metrics['Recall']:.4f}")

            with col2:
                st.metric("F1-score", f"{metrics['F1-score']:.4f}")
                st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")

            st.markdown("### 🔧 Best Parameters")
            if isinstance(best_params, dict):
                st.json(best_params)
            else:
                st.code(best_params)

            st.markdown("---")
            st.subheader("📊 Beneish M-Score (Baseline)")
            st.dataframe(pd.DataFrame([{
                "Accuracy": 0.8364,
                "Precision": 0.5556,
                "Recall": 0.5000,
                "F1-score": 0.5263,
                "ROC-AUC": 0.9044
            }]).round(4))

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👆 Upload an Excel file to start analysis.")
