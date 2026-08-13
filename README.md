# Hotel-Booking

Hotel management system that forecasts cancellation, revenue and allocates the
correct amount of staff — served as an interactive **Streamlit** web app.

## 🚀 Live app

The app (`app.py`) predicts the probability that a hotel booking is cancelled,
using your trained scikit-learn models. It also has a data-exploration tab and a
model-inspection tab.

## 📁 Project structure

```
Hotel-Booking/
├── app.py                  # Streamlit app (frontend + inference logic)
├── requirements.txt        # Libraries Streamlit Cloud installs
├── hotel_bookings.csv       # (optional) raw dataset for the EDA tab
├── .streamlit/config.toml  # Theme / server config
└── hotel_ai_models/        # Your exported models (.pkl / .joblib / .zip)
    ├── optimized_rf_cancellation.pkl
    ├── base_scaler.pkl
    └── ...
```

## ✅ What you need to add

This repo currently ships the **app scaffold only**. To make predictions, add
your exported artifacts (they weren't in the repo):

1. Put your model + scaler files in **`hotel_ai_models/`**. Loose `.pkl`,
   `.joblib`, `.sav`, or a `.zip` archive of them all work — the app
   auto-extracts zips at startup. See `hotel_ai_models/README.md` for details.
2. *(Optional)* Drop `hotel_bookings.csv` in the repo root to enable the EDA
   tab.

The app **introspects your models**: scikit-learn objects fitted on a pandas
DataFrame store `feature_names_in_`, and the app reads that to align the
sidebar inputs to the exact one-hot columns your `pd.get_dummies()` produced —
no manual column-mapping needed. Until models are present, the app runs in a
clearly-labelled demo mode instead of crashing.

## 🖥️ Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

A browser tab opens at http://localhost:8501.

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (already done for the app files).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **New app** → pick this repo/branch → main file path `app.py` → **Deploy**.
4. Every push to the branch auto-redeploys the app.

> **Version tip:** unpickling can break across scikit-learn major versions. If a
> `.pkl` fails to load on the cloud, pin the version you trained with in
> `requirements.txt` (e.g. `scikit-learn==1.5.1`).
