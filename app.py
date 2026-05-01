import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Student Productivity Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1,h2,h3,h4 {
    color: white;
}
div[data-testid="stMetric"] {
    background: #1c1f26;
    padding: 18px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    ann_model = load_model("models/ann_model.keras")
    ann_scaler = pickle.load(open("models/scaler.pkl", "rb"))

    linear_model = pickle.load(open("models/linear/model.pkl", "rb"))
    linear_scaler = pickle.load(open("models/linear/scaler.pkl", "rb"))

    multi_model = pickle.load(open("models/multilinear/model.pkl", "rb"))
    multi_scaler = pickle.load(open("models/multilinear/scaler.pkl", "rb"))

    poly_model = pickle.load(open("models/polynomial/model.pkl", "rb"))
    poly_scaler = pickle.load(open("models/polynomial/scaler.pkl", "rb"))
    poly_transform = pickle.load(open("models/polynomial/poly.pkl", "rb"))

    columns = pickle.load(open("models/columns.pkl", "rb"))

    return (
        ann_model, ann_scaler,
        linear_model, linear_scaler,
        multi_model, multi_scaler,
        poly_model, poly_scaler,
        poly_transform, columns
    )


(
ann_model, ann_scaler,
linear_model, linear_scaler,
multi_model, multi_scaler,
poly_model, poly_scaler,
poly_transform, columns
) = load_models()

# ==========================================================
# MODEL PERFORMANCE METRICS
# Replace with your exact values if needed
# ==========================================================

metrics = {
    "ANN": {
        "MSE": 0.006752,
        "MAE": 0.062671,
        "R2": 0.99997341,
        "Accuracy": 0.9421
    },
    "Linear": {
        "MSE": 0.000563,
        "MAE": 0.006813,
        "R2": 0.99999782,
        "Accuracy": 0.9930
    },
    "Multilinear": {
        "MSE": 0.000563,
        "MAE": 0.006813,
        "R2": 0.99999782,
        "Accuracy": 0.9930
    },
    "Polynomial": {
        "MSE": 0.000586,
        "MAE": 0.006964,
        "R2": 0.99999751,
        "Accuracy": 0.9935
    }
}

# ==========================================================
# RELIABILITY WEIGHT FUNCTION
# ==========================================================

def get_weight(m):
    return (
        m["R2"] +
        m["Accuracy"] +
        (1 / (1 + m["MAE"])) +
        (1 / (1 + m["MSE"]))
    ) / 4


# ==========================================================
# HEADER
# ==========================================================

st.title("📊 Student Productivity Prediction Dashboard")
st.markdown("### Compare 4 ML Models with Reliable Weighted Final Score")
st.divider()

# ==========================================================
# INPUT SECTION
# ==========================================================

st.sidebar.header("📝 Enter Student Attributes")

range_map = {
    "study_hours_per_day": (0.0, 24.0, 5.0),
    "focus_score": (0.0, 10.0, 5.0),
    "sleep_hours": (0.0, 12.0, 7.0),
    "attendance_percentage": (0.0, 100.0, 75.0),
    "stress_level": (0.0, 10.0, 5.0),
    "phone_usage_hours": (0.0, 24.0, 4.0),
    "social_media_hours": (0.0, 24.0, 3.0)
}

input_data = {}

for col in columns:

    if col in range_map:
        mn, mx, default = range_map[col]
    else:
        mn, mx, default = 0.0, 100.0, 50.0

    input_data[col] = st.sidebar.number_input(
        label=col,
        min_value=float(mn),
        max_value=float(mx),
        value=float(default),
        step=0.1
    )

input_df = pd.DataFrame([input_data])[columns]

# ==========================================================
# PREDICT BUTTON
# ==========================================================

if st.sidebar.button("🚀 Predict Productivity"):

    # ------------------------------------------------------
    # RAW PREDICTIONS
    # ------------------------------------------------------
    ann_pred = ann_model.predict(
        ann_scaler.transform(input_df),
        verbose=0
    )[0][0]

    lin_pred = linear_model.predict(
        linear_scaler.transform(input_df)
    )[0]

    multi_pred = multi_model.predict(
        multi_scaler.transform(input_df)
    )[0]

    poly_input = poly_transform.transform(input_df)

    poly_pred = poly_model.predict(
        poly_scaler.transform(poly_input)
    )[0]

    predictions = {
        "ANN": float(ann_pred),
        "Linear": float(lin_pred),
        "Multilinear": float(multi_pred),
        "Polynomial": float(poly_pred)
    }

    # ------------------------------------------------------
    # WEIGHTED FINAL SCORE (OUT OF 100)
    # ------------------------------------------------------
    
    weighted_sum = 0
    total_weight = 0
    weights = {}

    for model in predictions:
        w = get_weight(metrics[model])
        weights[model] = w
        weighted_sum += predictions[model] * w
        total_weight += w

    final_score = weighted_sum / total_weight

    # Clamp to 0 - 100
    final_score = max(0, min(final_score, 100))

    # ======================================================
    # MODEL PREDICTIONS
    # ======================================================
    
    st.subheader("🔮 Model Predictions (Out of 100)")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("ANN", f"{predictions['ANN']:.3f}")
    c2.metric("Linear", f"{predictions['Linear']:.3f}")
    c3.metric("Multilinear", f"{predictions['Multilinear']:.3f}")
    c4.metric("Polynomial", f"{predictions['Polynomial']:.3f}")

    st.divider()

    # ======================================================
    # BAR CHART
    # ======================================================
    
    st.subheader("📈 Prediction Comparison")

    chart_df = pd.DataFrame({
        "Model": list(predictions.keys()),
        "Prediction": list(predictions.values())
    })

    st.bar_chart(chart_df.set_index("Model"))

    st.divider()

    # ======================================================
    # METRICS TABLE
    # ======================================================
    
    st.subheader("📊 Model Reliability Comparison")

    rows = []

    for model in predictions:
        rows.append({
            "Model": model,
            "Prediction": round(predictions[model], 4),
            "Weight": round(weights[model], 6),
            "MSE": f"{metrics[model]['MSE']:.6f}",
            "MAE": f"{metrics[model]['MAE']:.6f}",
            "R²": f"{metrics[model]['R2']:.8f}",
            "Accuracy": f"{metrics[model]['Accuracy']*100:.2f}%"
        })

    table_df = pd.DataFrame(rows)
    st.dataframe(table_df, use_container_width=True)

    st.divider()

    # ======================================================
    # FINAL SCORE
    # ======================================================
    
    st.subheader("🎯 Final Reliable Productivity Score")

    progress_value = int(final_score)
    st.progress(progress_value)

    st.metric(
        label="Weighted Final Score",
        value=f"{final_score:.3f} / 100"
    )

    st.divider()

    # ======================================================
    # BEST MODEL
    # ======================================================
    
    best_model = max(weights, key=weights.get)

    st.subheader("🏆 Best Performing Model")
    st.success(best_model)

    st.divider()

    # ======================================================
    # INSIGHT
    # ======================================================
    
    st.subheader("📌 Insight")

    if final_score >= 85:
        st.success("Excellent productivity expected.")
    elif final_score >= 70:
        st.info("Good productivity level.")
    elif final_score >= 50:
        st.warning("Average productivity level.")
    else:
        st.error("Low productivity predicted.")
