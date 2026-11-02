# Diet Plan Model — Streamlit

## Files
- `diet_plan_model_fixed.py`: trains the models and saves the Random Forest artifacts (syntax error fixed).
- `app.py`: Streamlit prediction interface.
- `food_nutrition_dataset__2_.csv`: the dataset (must stay in the same folder).
- `requirements.txt`: Python dependencies.
- `best_random_forest_model.pkl`, `scaler.pkl`, `label_encoder.pkl`: trained model files (already included).

## Run locally
1. Install packages: `pip install -r requirements.txt`
2. (Optional) Retrain: `python diet_plan_model_fixed.py`
3. Launch: `streamlit run app.py`

## Streamlit Cloud
Upload ALL files in this folder to your GitHub repo (including the three `.pkl` files and the CSV), then deploy `app.py`.
