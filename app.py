import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "food_nutrition_dataset__2_.csv")

FEATURES = [
    "calories_kcal", "protein_g", "carbs_g", "fat_g", "saturated_fat_g",
    "fiber_g", "sugar_g", "caffeine_mg", "serving_size_g_or_ml",
]
LABELS = {
    "calories_kcal": "Calories (kcal)",
    "protein_g": "Protein (g)",
    "carbs_g": "Carbs (g)",
    "fat_g": "Fat (g)",
    "saturated_fat_g": "Saturated fat (g)",
    "fiber_g": "Fiber (g)",
    "sugar_g": "Sugar (g)",
    "caffeine_mg": "Caffeine (mg)",
    "serving_size_g_or_ml": "Serving size (g or ml)",
}

st.set_page_config(page_title="Diet Plan Model", page_icon="🥗", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "best_random_forest_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    return model, scaler, encoder


@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)


st.title("🥗 Diet Plan Model")
st.caption("Predict a food's category from its nutrition values, or browse the food dataset.")

try:
    model, scaler, encoder = load_artifacts()
except Exception as e:
    st.error(
        "Model files not found. Run `python diet_plan_model_fixed.py` first "
        f"to create them.\n\nDetails: {e}"
    )
    st.stop()

tab_predict, tab_browse = st.tabs(["🔮 Predict category", "📋 Browse foods"])

with tab_predict:
    st.subheader("Enter nutrition values")
    cols = st.columns(3)
    defaults = {
        "calories_kcal": 150.0, "protein_g": 10.0, "carbs_g": 20.0, "fat_g": 5.0,
        "saturated_fat_g": 1.0, "fiber_g": 2.0, "sugar_g": 3.0,
        "caffeine_mg": 0.0, "serving_size_g_or_ml": 100.0,
    }
    values = {}
    for i, f in enumerate(FEATURES):
        with cols[i % 3]:
            values[f] = st.number_input(LABELS[f], min_value=0.0, value=defaults[f], step=1.0)

    if st.button("Predict category", type="primary"):
        x = pd.DataFrame([values], columns=FEATURES)
        x_scaled = scaler.transform(x)
        proba = model.predict_proba(x_scaled)[0]
        top = np.argsort(proba)[::-1][:3]
        best_name = encoder.inverse_transform([model.classes_[top[0]]])[0]
        st.success(f"Predicted category: **{best_name}** ({proba[top[0]] * 100:.1f}% confidence)")
        st.write("Top 3 guesses:")
        st.table(pd.DataFrame({
            "Category": encoder.inverse_transform(model.classes_[top]),
            "Confidence": [f"{proba[i] * 100:.1f}%" for i in top],
        }))

with tab_browse:
    try:
        df = load_data()
        c1, c2 = st.columns([1, 2])
        with c1:
            cats = ["All"] + sorted(df["category"].dropna().unique().tolist())
            cat = st.selectbox("Category", cats)
        with c2:
            q = st.text_input("Search food name")
        view = df.copy()
        if cat != "All":
            view = view[view["category"] == cat]
        if q:
            view = view[view["food_name"].str.contains(q, case=False, na=False)]
        st.write(f"{len(view)} foods")
        st.dataframe(view, width="stretch", hide_index=True)
    except FileNotFoundError:
        st.warning("Dataset CSV not found. Keep it in the same folder as app.py.")
