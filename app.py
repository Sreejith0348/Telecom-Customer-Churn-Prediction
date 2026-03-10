import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report, precision_recall_curve
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a2f3e; }
    .metric-card {
        background: #161b27;
        border: 1px solid #2a2f3e;
        border-radius: 10px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-label { color: #8b92a5; font-size: 13px; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
    .metric-value { color: #e2e8f0; font-size: 32px; font-weight: 700; line-height: 1; }
    .metric-delta-pos { color: #4ade80; font-size: 13px; margin-top: 4px; }
    .metric-delta-neg { color: #f87171; font-size: 13px; margin-top: 4px; }
    .section-header {
        color: #e2e8f0; font-size: 18px; font-weight: 600;
        border-left: 3px solid #6366f1; padding-left: 12px;
        margin: 24px 0 16px 0;
    }
    .stSelectbox label, .stSlider label, .stMultiSelect label { color: #8b92a5 !important; font-size: 13px !important; }
    div[data-testid="stHorizontalBlock"] { gap: 16px; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#6366f1", "#f87171", "#4ade80", "#facc15", "#38bdf8", "#e879f9"]
PLOTLY_THEME = dict(
    paper_bgcolor="#161b27",
    plot_bgcolor="#161b27",
    font_color="#8b92a5",
    title_font_color="#e2e8f0",
    legend_bgcolor="#161b27",
    legend_bordercolor="#2a2f3e",
)

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df.drop("customerID", axis=1, inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    # Derived features
    df["ChargePerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["HasMultipleServices"] = (
        (df["PhoneService"] == "Yes").astype(int) +
        (df["OnlineSecurity"] == "Yes").astype(int) +
        (df["OnlineBackup"] == "Yes").astype(int) +
        (df["DeviceProtection"] == "Yes").astype(int) +
        (df["TechSupport"] == "Yes").astype(int) +
        (df["StreamingTV"] == "Yes").astype(int) +
        (df["StreamingMovies"] == "Yes").astype(int)
    )
    return df


@st.cache_data
def build_model(df: pd.DataFrame, model_name: str, test_size: float, use_smote: bool):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    if use_smote:
        sm = SMOTE(random_state=42)
        X_train_proc, y_train = sm.fit_resample(X_train_proc, y_train)

    models_map = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    model = models_map[model_name]
    model.fit(X_train_proc, y_train)

    y_pred       = model.predict(X_test_proc)
    y_pred_proba = model.predict_proba(X_test_proc)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_pred_proba)
    cm      = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)

    # Feature importance
    feature_names = (
        num_features +
        preprocessor.named_transformers_["cat"]
            .named_steps["encoder"]
            .get_feature_names_out(cat_features).tolist()
    )
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        perm = permutation_importance(model, X_test_proc, y_test, n_repeats=5, random_state=42)
        importances = perm.importances_mean

    feat_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(15)

    return {
        "model": model,
        "acc": acc, "auc": auc,
        "cm": cm,
        "fpr": fpr, "tpr": tpr,
        "prec": prec, "rec": rec,
        "y_pred_proba": y_pred_proba,
        "y_test": y_test.values,
        "feat_df": feat_df,
        "X_train_size": X_train_proc.shape[0],
        "X_test_size": X_test_proc.shape[0],
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## Configuration")
st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
default_path = "data/Telco-Customer-Churn.csv"

if uploaded:
    df_raw = load_data(uploaded)
else:
    try:
        df_raw = load_data(default_path)
    except FileNotFoundError:
        st.error("Place Telco-Customer-Churn.csv in the same folder, or upload it via the sidebar.")
        st.stop()

st.sidebar.markdown("### Model Settings")
model_choice = st.sidebar.selectbox(
    "Algorithm",
    ["Gradient Boosting", "Random Forest", "Logistic Regression"],
)
test_split = st.sidebar.slider("Test split size", 0.1, 0.4, 0.2, 0.05)
use_smote  = st.sidebar.checkbox("Apply SMOTE balancing", value=True)

st.sidebar.markdown("### Explorer Filters")
contract_filter = st.sidebar.multiselect(
    "Contract type",
    df_raw["Contract"].unique().tolist(),
    default=df_raw["Contract"].unique().tolist(),
)
internet_filter = st.sidebar.multiselect(
    "Internet service",
    df_raw["InternetService"].unique().tolist(),
    default=df_raw["InternetService"].unique().tolist(),
)
tenure_range = st.sidebar.slider(
    "Tenure (months)",
    int(df_raw["tenure"].min()),
    int(df_raw["tenure"].max()),
    (int(df_raw["tenure"].min()), int(df_raw["tenure"].max())),
)

df = df_raw[
    df_raw["Contract"].isin(contract_filter) &
    df_raw["InternetService"].isin(internet_filter) &
    df_raw["tenure"].between(*tenure_range)
].copy()

# ─────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────
with st.spinner("Training model..."):
    results = build_model(df, model_choice, test_split, use_smote)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#e2e8f0;font-size:28px;font-weight:700;margin-bottom:4px;'>"
    "Telco Customer Churn Dashboard</h1>"
    "<p style='color:#8b92a5;font-size:14px;margin-bottom:24px;'>"
    f"Model: {model_choice} &nbsp;|&nbsp; {len(df):,} customers &nbsp;|&nbsp; "
    f"Train: {results['X_train_size']:,} &nbsp;|&nbsp; Test: {results['X_test_size']:,}"
    "</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_overview, tab_eda, tab_model, tab_predictor = st.tabs([
    "Overview", "Data Explorer", "Model Performance", "Predict Customer"
])

# ═══════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════
with tab_overview:
    churn_rate  = df["Churn"].mean() * 100
    avg_tenure  = df["tenure"].mean()
    avg_monthly = df["MonthlyCharges"].mean()
    churned_n   = df["Churn"].sum()

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Total Customers",    f"{len(df):,}",       None, None),
        (c2, "Churn Rate",         f"{churn_rate:.1f}%",  churn_rate > 27, "High" if churn_rate > 27 else "Normal"),
        (c3, "Avg Monthly Charge", f"${avg_monthly:.0f}", None, None),
        (c4, "Avg Tenure",         f"{avg_tenure:.0f} mo", None, None),
    ]
    for col, label, value, is_neg, note in cards:
        with col:
            delta_html = ""
            if note:
                cls = "metric-delta-neg" if is_neg else "metric-delta-pos"
                delta_html = f'<div class="{cls}">{note}</div>'
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'{delta_html}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown('<div class="section-header">Churn Distribution</div>', unsafe_allow_html=True)
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Label"] = churn_counts["Churn"].map({1: "Churned", 0: "Retained"})
        fig = px.pie(
            churn_counts, values="Count", names="Label",
            color_discrete_sequence=[PALETTE[1], PALETTE[0]],
            hole=0.55,
        )
        fig.update_layout(**PLOTLY_THEME, margin=dict(t=20, b=20, l=20, r=20), height=300)
        fig.update_traces(textfont_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Churn by Contract Type</div>', unsafe_allow_html=True)
        contract_churn = df.groupby("Contract")["Churn"].mean().reset_index()
        contract_churn.columns = ["Contract", "ChurnRate"]
        contract_churn["ChurnRate"] *= 100
        fig = px.bar(
            contract_churn, x="Contract", y="ChurnRate",
            color="Contract", color_discrete_sequence=PALETTE,
            labels={"ChurnRate": "Churn Rate (%)"},
        )
        fig.update_layout(**PLOTLY_THEME, showlegend=False, margin=dict(t=20, b=20), height=300)
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Tenure Distribution by Churn</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x="tenure", color=df["Churn"].map({1: "Churned", 0: "Retained"}),
            barmode="overlay", nbins=40,
            color_discrete_map={"Churned": PALETTE[1], "Retained": PALETTE[0]},
            labels={"color": "Status", "tenure": "Tenure (months)"},
        )
        fig.update_layout(**PLOTLY_THEME, margin=dict(t=20, b=20), height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Monthly Charges by Churn</div>', unsafe_allow_html=True)
        fig = px.box(
            df, x=df["Churn"].map({1: "Churned", 0: "Retained"}),
            y="MonthlyCharges",
            color=df["Churn"].map({1: "Churned", 0: "Retained"}),
            color_discrete_map={"Churned": PALETTE[1], "Retained": PALETTE[0]},
            labels={"x": "Status", "MonthlyCharges": "Monthly Charges ($)"},
        )
        fig.update_layout(**PLOTLY_THEME, showlegend=False, margin=dict(t=20, b=20), height=280)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 2 — DATA EXPLORER
# ═══════════════════════════════════════════
with tab_eda:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown('<div class="section-header">Churn Rate by Category</div>', unsafe_allow_html=True)
        cat_cols = ["InternetService", "PaymentMethod", "Contract",
                    "TechSupport", "OnlineSecurity", "StreamingTV"]
        selected_cat = st.selectbox("Select feature", cat_cols, key="eda_cat")
        grp = df.groupby(selected_cat)["Churn"].mean().reset_index()
        grp["Churn"] *= 100
        grp = grp.sort_values("Churn", ascending=True)
        fig = px.bar(
            grp, x="Churn", y=selected_cat, orientation="h",
            color="Churn", color_continuous_scale=["#6366f1", "#f87171"],
            labels={"Churn": "Churn Rate (%)"},
        )
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False,
                          margin=dict(t=20, b=20), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Numeric Feature vs Churn</div>', unsafe_allow_html=True)
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "ChargePerMonth", "HasMultipleServices"]
        selected_num = st.selectbox("Select feature", num_cols, key="eda_num")
        fig = px.violin(
            df, y=selected_num, x=df["Churn"].map({1: "Churned", 0: "Retained"}),
            color=df["Churn"].map({1: "Churned", 0: "Retained"}),
            box=True, points=False,
            color_discrete_map={"Churned": PALETTE[1], "Retained": PALETTE[0]},
            labels={"x": "Status", selected_num: selected_num},
        )
        fig.update_layout(**PLOTLY_THEME, showlegend=False,
                          margin=dict(t=20, b=20), height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Scatter: Tenure vs Monthly Charges</div>', unsafe_allow_html=True)
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="tenure", y="MonthlyCharges",
        color=sample["Churn"].map({1: "Churned", 0: "Retained"}),
        color_discrete_map={"Churned": PALETTE[1], "Retained": PALETTE[0]},
        opacity=0.55,
        labels={"color": "Status"},
        marginal_x="histogram", marginal_y="histogram",
    )
    fig.update_layout(**PLOTLY_THEME, height=420, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Raw Data Sample</div>', unsafe_allow_html=True)
    st.dataframe(
        df.head(200).style.applymap(
            lambda v: "color: #f87171" if v == 1 else "", subset=["Churn"]
        ),
        use_container_width=True,
        height=280,
    )


# ═══════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════
with tab_model:
    # Top metrics
    tn, fp, fn, tp = results["cm"].ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1          = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label, val in [
        (m1, "Accuracy",    f"{results['acc']:.3f}"),
        (m2, "ROC-AUC",     f"{results['auc']:.3f}"),
        (m3, "Sensitivity", f"{sensitivity:.3f}"),
        (m4, "Specificity", f"{specificity:.3f}"),
        (m5, "F1 Score",    f"{f1:.3f}"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    row1_l, row1_r = st.columns(2)

    with row1_l:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results["fpr"], y=results["tpr"], mode="lines",
            line=dict(color=PALETTE[0], width=2.5),
            name=f"AUC = {results['auc']:.3f}",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#4b5563", width=1.5, dash="dash"),
            showlegend=False,
        ))
        fig.update_layout(
            **PLOTLY_THEME,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=320, margin=dict(t=20, b=40, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with row1_r:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_labels = [["TN", "FP"], ["FN", "TP"]]
        z_text = [[f"{cm_labels[i][j]}<br>{results['cm'][i][j]:,}" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(
            z=results["cm"],
            x=["Predicted: No Churn", "Predicted: Churn"],
            y=["Actual: No Churn", "Actual: Churn"],
            colorscale=[[0, "#1e2535"], [1, "#6366f1"]],
            text=z_text, texttemplate="%{text}",
            textfont=dict(color="#e2e8f0", size=14),
            showscale=False,
        ))
        fig.update_layout(
            **PLOTLY_THEME,
            height=320, margin=dict(t=20, b=40, l=80, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    row2_l, row2_r = st.columns(2)

    with row2_l:
        st.markdown('<div class="section-header">Feature Importance (Top 15)</div>', unsafe_allow_html=True)
        feat_df = results["feat_df"].sort_values("importance")
        fig = px.bar(
            feat_df, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale=["#4338ca", "#6366f1", "#a5b4fc"],
            labels={"importance": "Importance", "feature": ""},
        )
        fig.update_layout(
            **PLOTLY_THEME, coloraxis_showscale=False,
            height=360, margin=dict(t=20, b=20, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with row2_r:
        st.markdown('<div class="section-header">Predicted Probability Distribution</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Probability": results["y_pred_proba"],
            "Actual": ["Churned" if y == 1 else "Retained" for y in results["y_test"]],
        })
        fig = px.histogram(
            prob_df, x="Probability", color="Actual",
            nbins=50, barmode="overlay", opacity=0.75,
            color_discrete_map={"Churned": PALETTE[1], "Retained": PALETTE[0]},
            labels={"Probability": "Predicted Churn Probability"},
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="#facc15", annotation_text="Threshold 0.5")
        fig.update_layout(
            **PLOTLY_THEME,
            height=360, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Precision-Recall Curve</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["rec"], y=results["prec"], mode="lines",
        line=dict(color=PALETTE[3], width=2.5),
        fill="tozeroy", fillcolor="rgba(250,204,21,0.08)",
        name="Precision-Recall",
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        xaxis_title="Recall", yaxis_title="Precision",
        height=280, margin=dict(t=20, b=40, l=40, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 4 — SINGLE CUSTOMER PREDICTOR
# ═══════════════════════════════════════════
with tab_predictor:
    st.markdown('<div class="section-header">Predict Churn for a Single Customer</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#8b92a5;font-size:13px;'>Fill in customer details to get a real-time churn prediction.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        gender           = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen   = st.selectbox("Senior Citizen", [0, 1])
        partner          = st.selectbox("Partner", ["Yes", "No"])
        dependents       = st.selectbox("Dependents", ["Yes", "No"])
        tenure           = st.slider("Tenure (months)", 0, 72, 12)
        phone_service    = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines   = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security  = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup    = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_prot      = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support     = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv     = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        contract         = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless        = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method   = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
        total_charges    = st.slider("Total Charges ($)", 0.0, 9000.0, float(monthly_charges * tenure), 10.0)

    if st.button("Run Prediction", type="primary"):
        # Build input row matching training columns (minus derived, which are added)
        input_dict = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "ChargePerMonth": total_charges / (tenure + 1),
            "HasMultipleServices": sum([
                phone_service == "Yes",
                online_security == "Yes",
                online_backup == "Yes",
                device_prot == "Yes",
                tech_support == "Yes",
                streaming_tv == "Yes",
                streaming_movies == "Yes",
            ]),
        }
        input_df = pd.DataFrame([input_dict])

        # Rebuild preprocessor on full df to match training schema
        X_full = df.drop("Churn", axis=1)
        y_full = df["Churn"]
        num_f = X_full.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_f = X_full.select_dtypes(include=["object"]).columns.tolist()
        prep = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", MinMaxScaler())]), num_f),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("enc", OneHotEncoder(handle_unknown="ignore", drop="first"))]), cat_f),
        ])
        prep.fit(X_full)
        input_proc = prep.transform(input_df)
        prob = results["model"].predict_proba(input_proc)[0][1]

        # Display
        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns([1, 1, 2])

        color = "#f87171" if prob >= 0.5 else "#4ade80"
        verdict = "LIKELY TO CHURN" if prob >= 0.5 else "LIKELY TO STAY"
        with res_col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Churn Probability</div>'
                f'<div class="metric-value" style="color:{color}">{prob:.1%}</div>'
                f'<div style="color:{color};font-size:13px;margin-top:6px;">{verdict}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with res_col2:
            risk = "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")
            risk_color = "#f87171" if risk == "High" else ("#facc15" if risk == "Medium" else "#4ade80")
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Risk Level</div>'
                f'<div class="metric-value" style="color:{risk_color}">{risk}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with res_col3:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"color": color, "size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#4b5563"},
                    "bar": {"color": color},
                    "bgcolor": "#1e2535",
                    "steps": [
                        {"range": [0, 40],  "color": "#1e2535"},
                        {"range": [40, 70], "color": "#292e40"},
                        {"range": [70, 100],"color": "#2d2035"},
                    ],
                    "threshold": {"line": {"color": "#facc15", "width": 3}, "value": 50},
                },
            ))
            fig.update_layout(
                paper_bgcolor="#161b27", font_color="#8b92a5",
                height=200, margin=dict(t=20, b=10, l=30, r=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Retention tips
        if prob >= 0.5:
            st.markdown('<div class="section-header">Recommended Retention Actions</div>', unsafe_allow_html=True)
            tips = []
            if contract == "Month-to-month":
                tips.append("Offer a discounted annual or two-year contract to increase commitment.")
            if internet_service == "Fiber optic" and monthly_charges > 80:
                tips.append("Fiber optic with high charges is a strong churn driver — consider a loyalty discount.")
            if online_security == "No":
                tips.append("Promote the Online Security add-on — customers without it churn more often.")
            if tech_support == "No":
                tips.append("Offer a free trial of Tech Support to increase service stickiness.")
            if tenure < 12:
                tips.append("Early-tenure customers are at high risk — consider an onboarding loyalty reward.")
            if not tips:
                tips.append("Reach out proactively with a personalized offer or satisfaction check-in.")
            for tip in tips:
                st.markdown(f"- {tip}")