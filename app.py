"""
Hotel Booking AI — Streamlit web app
====================================

An interactive web app that serves a hotel-booking dataset and trained
scikit-learn models (e.g. an optimized Random Forest cancellation predictor)
online via Streamlit Community Cloud.

Design notes
------------
The app is deliberately *robust to your exact preprocessing*. Instead of
hard-coding the list of one-hot columns produced by ``pd.get_dummies()`` during
training, it reads the columns straight off your fitted objects:

* A fitted ``StandardScaler`` (or any sklearn transformer/estimator trained on a
  DataFrame) exposes ``feature_names_in_`` — the exact ordered column names it
  was fit on. The app uses that as the source of truth.
* User inputs are collected as *raw* human-friendly fields, expanded into
  one-hot dummy columns, then re-indexed onto the model's expected columns
  (missing columns filled with 0). This mirrors what ``get_dummies`` produced at
  training time, regardless of which categories or drop_first setting you used.

Just drop your exported files into ``hotel_ai_models/`` (``.pkl`` files, or
``.zip`` archives containing them — the app auto-extracts zips) and the app
adapts automatically. If nothing is found, the app runs in a clearly-labelled
demo mode so the deployment never crashes.
"""

from __future__ import annotations

import datetime as _dt
import glob
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except Exception:  # pragma: no cover - joblib is in requirements.txt
    joblib = None


# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "hotel_ai_models"
DATA_CANDIDATES = [
    APP_DIR / "hotel_bookings.csv",
    APP_DIR / "hotel_booking.csv",
    APP_DIR / "data" / "hotel_bookings.csv",
]

# Keywords used to auto-classify loaded objects.
SCALER_HINTS = ("scaler", "standardscaler", "minmax")
MODEL_HINTS = ("rf", "randomforest", "random_forest", "model", "clf",
               "classifier", "xgb", "gboost", "logistic", "cancellation")

# Month ordering for the arrival-month dropdown.
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


# --------------------------------------------------------------------------- #
# 1. Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Hotel AI Manager",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# 2. Model / scaler loading
# --------------------------------------------------------------------------- #
def _extract_zip_archives(models_dir: Path) -> None:
    """Extract any ``*.zip`` archives in ``models_dir`` in place.

    Users may commit their models as zip archives (smaller, and avoids Git LFS
    for larger forests). We unzip them next to the archive so the ``.pkl`` files
    become available. Extraction is skipped when the target files already exist.
    """
    for zip_path in models_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                members = [m for m in zf.namelist() if not m.endswith("/")]
                needs_extract = any(
                    not (models_dir / Path(m).name).exists() for m in members
                )
                if not needs_extract:
                    continue
                for member in members:
                    # Flatten any internal directory structure into models_dir.
                    target = models_dir / Path(member).name
                    if target.exists():
                        continue
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
        except zipfile.BadZipFile:
            st.warning(f"Could not read zip archive: {zip_path.name}")


@st.cache_resource(show_spinner="Loading models…")
def load_artifacts() -> dict:
    """Discover, extract and load every model/scaler under ``hotel_ai_models/``.

    Returns a dict with:
        ``objects``  -> {name: loaded_object}
        ``scalers``  -> {name: object}   (subset classified as scalers)
        ``models``   -> {name: object}   (subset classified as estimators)
        ``errors``   -> list[str]
    """
    result = {"objects": {}, "scalers": {}, "models": {}, "errors": []}

    if not MODELS_DIR.exists():
        return result
    if joblib is None:
        result["errors"].append("joblib is not installed.")
        return result

    _extract_zip_archives(MODELS_DIR)

    pkl_files = sorted(
        glob.glob(str(MODELS_DIR / "*.pkl"))
        + glob.glob(str(MODELS_DIR / "*.joblib"))
        + glob.glob(str(MODELS_DIR / "*.sav"))
    )

    for path in pkl_files:
        name = Path(path).stem
        try:
            obj = joblib.load(path)
        except Exception as exc:  # pragma: no cover - depends on user files
            result["errors"].append(f"{Path(path).name}: {exc}")
            continue

        result["objects"][name] = obj
        low = name.lower()

        is_scaler = any(h in low for h in SCALER_HINTS) or (
            hasattr(obj, "transform")
            and not hasattr(obj, "predict")
            and hasattr(obj, "mean_")
        )
        is_model = hasattr(obj, "predict")

        if is_model:
            result["models"][name] = obj
        elif is_scaler:
            result["scalers"][name] = obj
        else:
            # Unclassified transformer — expose as a scaler-like object.
            if hasattr(obj, "transform"):
                result["scalers"][name] = obj

    return result


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame | None:
    """Load the raw dataset for the EDA tab, if present."""
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            try:
                return pd.read_csv(candidate)
            except Exception:
                return None
    return None


def expected_features(scaler, model) -> list[str] | None:
    """Return the ordered feature names the pipeline expects, or ``None``.

    Preference order: the *model's* fitted columns (the model is the final
    consumer), then the scaler's. For a correctly-paired model/scaler these are
    identical; preferring the model keeps us right even if a scaler is missing.
    """
    for obj in (model, scaler):
        if obj is not None and hasattr(obj, "feature_names_in_"):
            return list(obj.feature_names_in_)
    # Fall back to feature count only (names unknown).
    return None


def pick_scaler(model, scalers: dict, model_name: str = ""):
    """Choose the scaler that matches this model, or ``None``.

    Different model families here have different feature spaces (cancellation vs
    upsell vs pricing).

    1. If the model carries ``feature_names_in_`` (trees, RF), pair it with the
       scaler whose columns overlap ~completely — otherwise scale nothing (the
       right choice for the tree pricing regressor trained unscaled).
    2. If the model has *no* feature names (e.g. LogisticRegression trained on a
       scaled array), fall back to matching by feature count, disambiguating by
       name family (``upsell`` vs cancellation ``base``).
    """
    if not scalers:
        return None

    if hasattr(model, "feature_names_in_"):
        model_cols = set(model.feature_names_in_)
        for sc in scalers.values():
            if not hasattr(sc, "feature_names_in_"):
                continue
            # Require an exact column-set match — a scaler with *nearly* the same
            # columns (e.g. the 239-col cancellation scaler vs the 222-col pricing
            # model) would silently reshape the input to the wrong width.
            if set(sc.feature_names_in_) == model_cols:
                return sc
        return None

    # No feature names: match by count, then by name family.
    n = getattr(model, "n_features_in_", None)
    candidates = [
        (nm, sc) for nm, sc in scalers.items()
        if getattr(sc, "n_features_in_", None) == n
    ] or list(scalers.items())
    if len(candidates) == 1:
        return candidates[0][1]
    want_upsell = "upsell" in model_name.lower()
    for nm, sc in candidates:
        if ("upsell" in nm.lower()) == want_upsell:
            return sc
    return candidates[0][1]


def country_codes(artifacts: dict) -> list[str]:
    """ISO codes the loaded models recognise, from their ``country_*`` columns."""
    codes: set[str] = set()
    for obj in artifacts["objects"].values():
        names = getattr(obj, "feature_names_in_", None)
        if names is None:
            continue
        for col in names:
            if isinstance(col, str) and col.startswith("country_"):
                codes.add(col[len("country_"):])
    ordered = sorted(codes)
    if "PRT" in ordered:  # most common origin in this dataset — surface it first
        ordered = ["PRT"] + [c for c in ordered if c != "PRT"]
    return ordered


# --------------------------------------------------------------------------- #
# 3. Feature engineering — map raw UI inputs to the model's encoded columns
# --------------------------------------------------------------------------- #
def collect_raw_inputs(country_options: list[str] | None = None) -> dict:
    """Render sidebar widgets and return a dict of raw (un-encoded) inputs.

    Fields mirror the well-known *Hotel Booking Demand* dataset. Not every model
    will use every field — unused ones are simply dropped during re-indexing.

    ``country_options`` is the list of ISO country codes the loaded models
    actually recognise (derived from their ``country_*`` dummy columns), so the
    dropdown only ever offers codes that map to a real feature.
    """
    st.sidebar.header("🧾 Guest & Booking Details")

    with st.sidebar.expander("Stay", expanded=True):
        hotel = st.selectbox("Hotel type", ["City Hotel", "Resort Hotel"])
        lead_time = st.slider("Lead time (days before arrival)", 0, 737, 30)
        arrival_year = st.selectbox("Arrival year", [2015, 2016, 2017], index=1)
        arrival_month = st.selectbox("Arrival month", MONTHS, index=6)
        arrival_day = st.slider("Arrival day of month", 1, 31, 15)
        week_nights = st.slider("Week nights", 0, 30, 2)
        weekend_nights = st.slider("Weekend nights", 0, 16, 1)

    with st.sidebar.expander("Party", expanded=True):
        adults = st.number_input("Adults", 1, 10, 2)
        children = st.number_input("Children", 0, 10, 0)
        babies = st.number_input("Babies", 0, 10, 0)

    with st.sidebar.expander("Booking", expanded=True):
        market_segment = st.selectbox(
            "Market segment",
            ["Online TA", "Offline TA/TO", "Direct", "Corporate",
             "Groups", "Complementary", "Aviation", "Undefined"],
        )
        distribution_channel = st.selectbox(
            "Distribution channel",
            ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
        )
        deposit_type = st.selectbox(
            "Deposit type", ["No Deposit", "Non Refund", "Refundable"]
        )
        customer_type = st.selectbox(
            "Customer type",
            ["Transient", "Transient-Party", "Contract", "Group"],
        )
        meal = st.selectbox("Meal", ["BB", "HB", "FB", "SC", "Undefined"])
        reserved_room_type = st.selectbox(
            "Reserved room type",
            list("ABCDEFGHLP"),
        )
        opts = country_options or ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA",
                                   "IRL", "BEL", "BRA", "NLD", "USA", "CHE"]
        default_country = opts.index("PRT") if "PRT" in opts else 0
        country = st.selectbox("Country (guest origin)", opts, index=default_country)
        agent = st.number_input("Agent ID (0 = none)", 0, 600, 9)

    with st.sidebar.expander("History & extras", expanded=False):
        is_repeated_guest = st.selectbox("Repeated guest?", ["No", "Yes"])
        previous_cancellations = st.number_input(
            "Previous cancellations", 0, 30, 0
        )
        previous_bookings_not_canceled = st.number_input(
            "Previous non-cancelled bookings", 0, 100, 0
        )
        booking_changes = st.number_input("Booking changes", 0, 30, 0)
        days_in_waiting_list = st.number_input("Days in waiting list", 0, 500, 0)
        adr = st.number_input("ADR (avg daily rate)", 0.0, 6000.0, 100.0, step=5.0)
        required_car_parking_spaces = st.number_input("Parking spaces", 0, 8, 0)
        total_of_special_requests = st.number_input("Special requests", 0, 10, 0)
        is_canceled = st.selectbox(
            "Booking currently cancelled? (upsell model only)", ["No", "Yes"]
        )

    month_num = MONTHS.index(arrival_month) + 1
    total_nights = week_nights + weekend_nights
    total_guests = adults + children + babies
    try:
        week_number = _dt.date(arrival_year, month_num, arrival_day).isocalendar()[1]
    except ValueError:
        week_number = min(53, (month_num - 1) * 4 + 2)

    return {
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_year,
        "arrival_date_month": arrival_month,
        "arrival_date_month_num": month_num,
        "arrival_date_week_number": week_number,
        "arrival_date_day_of_month": arrival_day,
        "stays_in_week_nights": week_nights,
        "stays_in_weekend_nights": weekend_nights,
        "total_nights": total_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "total_guests": total_guests,
        "country": country,
        "agent": agent,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
        "meal": meal,
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": reserved_room_type,
        "is_repeated_guest": 1 if is_repeated_guest == "Yes" else 0,
        "is_canceled": 1 if is_canceled == "Yes" else 0,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "booking_changes": booking_changes,
        "days_in_waiting_list": days_in_waiting_list,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
    }


# Which raw fields are categorical (get one-hot expanded) vs numeric.
_CATEGORICAL = {
    "hotel", "arrival_date_month", "market_segment", "distribution_channel",
    "deposit_type", "customer_type", "meal", "reserved_room_type",
    "assigned_room_type", "country",
}

# Helper keys that are not model features (used only to derive other values).
_NON_FEATURE = {"arrival_date_month_num"}


def build_feature_row(raw: dict, expected_cols: list[str] | None) -> pd.DataFrame:
    """Turn raw inputs into a single-row DataFrame aligned to ``expected_cols``.

    Strategy:
        1. Split raw inputs into numeric and categorical.
        2. One-hot encode the categoricals the same way ``pd.get_dummies`` does,
           producing columns named ``<field>_<value>``.
        3. Re-index onto ``expected_cols`` (drop extras, add missing as 0). This
           makes the row match the training design matrix exactly, whatever the
           original category set / drop_first choice was.

    When ``expected_cols`` is unknown (models don't carry ``feature_names_in_``),
    we return the full encoded frame and let the caller handle alignment by
    count.
    """
    numeric = {k: v for k, v in raw.items()
               if k not in _CATEGORICAL and k not in _NON_FEATURE
               and not isinstance(v, str)}
    frame = pd.DataFrame([numeric])

    for field in _CATEGORICAL:
        if field in raw:
            value = raw[field]
            dummies = pd.get_dummies(
                pd.Series([value], name=field), prefix=field
            )
            frame = pd.concat([frame.reset_index(drop=True), dummies], axis=1)

    if expected_cols is not None:
        frame = frame.reindex(columns=expected_cols, fill_value=0)

    return frame.astype(float)


def run_prediction(model, scaler, features: pd.DataFrame) -> dict:
    """Scale (if a scaler is paired) and predict.

    Returns a dict describing the result:
        {"kind": "classifier", "label": int, "proba": float}   or
        {"kind": "regressor",  "value": float}
    """
    X = features
    if scaler is not None and hasattr(scaler, "transform"):
        # Align to the scaler's own column order before transforming.
        if hasattr(scaler, "feature_names_in_"):
            X = features.reindex(
                columns=list(scaler.feature_names_in_), fill_value=0
            )
        X = scaler.transform(X)

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
        return {"kind": "classifier", "label": int(proba > 0.5), "proba": proba}
    return {"kind": "regressor", "value": float(model.predict(X)[0])}


# --------------------------------------------------------------------------- #
# 4. UI — header + tabs
# --------------------------------------------------------------------------- #
st.title("🏨 Hotel Booking AI Manager")
st.markdown(
    "Run the deployed models — **cancellation risk**, **dynamic pricing**, and "
    "**upsell** — explore the dataset, and inspect each model. Served from your "
    "GitHub repo via Streamlit Community Cloud."
)

artifacts = load_artifacts()
dataset = load_dataset()

predict_tab, data_tab, models_tab = st.tabs(
    ["🔮 Predict", "📊 Data & EDA", "🧠 Model Info"]
)


def _model_purpose(name: str) -> str:
    """Best-effort human label for what a model predicts, from its filename."""
    low = name.lower()
    if "pricing" in low or "regress" in low or "adr" in low:
        return "price"          # regression → predicted ADR
    if "upsell" in low:
        return "upsell"         # classification → upsell likelihood
    return "cancellation"       # default → cancellation risk


# ---- Predict tab ---------------------------------------------------------- #
with predict_tab:
    raw_inputs = collect_raw_inputs(country_codes(artifacts))

    model_names = list(artifacts["models"].keys())

    if not model_names:
        st.warning(
            "**Demo mode** — no models found in `hotel_ai_models/`.\n\n"
            "Add your exported files (`.pkl`, `.joblib`, or a `.zip` containing "
            "them) to the **`hotel_ai_models/`** folder and push to GitHub. "
            "Streamlit Cloud will redeploy automatically and this app will pick "
            "them up. Expected examples:\n"
            "- `optimized_rf_cancellation.pkl` (the classifier)\n"
            "- `base_scaler.pkl` (the fitted `StandardScaler`)"
        )
        st.divider()

    chosen_model = st.selectbox(
        "Model", model_names or ["(demo — no model loaded)"],
        help="Pick which trained estimator to run. The matching scaler is "
             "selected automatically.",
    )

    # Show which scaler will be paired with the chosen model.
    if model_names:
        _paired = pick_scaler(artifacts["models"][chosen_model],
                              artifacts["scalers"], chosen_model)
        _paired_name = next(
            (n for n, s in artifacts["scalers"].items() if s is _paired), None
        )
        st.caption(
            f"Purpose: **{_model_purpose(chosen_model)}** · "
            f"Scaler: **{_paired_name or 'none (unscaled)'}**"
        )

    predict_clicked = st.button(
        "Predict", type="primary", use_container_width=True
    )

    if predict_clicked:
        purpose = _model_purpose(chosen_model) if model_names else "cancellation"

        if not model_names:
            # Deterministic heuristic so the demo UI still responds sensibly.
            score = (
                0.35
                + 0.0004 * raw_inputs["lead_time"]
                + (0.25 if raw_inputs["deposit_type"] == "Non Refund" else 0)
                + (0.15 if raw_inputs["market_segment"] == "Groups" else 0)
                - 0.10 * raw_inputs["total_of_special_requests"]
                - 0.15 * raw_inputs["is_repeated_guest"]
            )
            result = {"kind": "classifier", "label": int(score > 0.5),
                      "proba": float(np.clip(score, 0.02, 0.98))}
            st.caption("Heuristic estimate (demo mode — no model loaded).")
        else:
            model = artifacts["models"][chosen_model]
            scaler = pick_scaler(model, artifacts["scalers"], chosen_model)
            exp_cols = expected_features(scaler, model)
            features = build_feature_row(raw_inputs, exp_cols)
            try:
                result = run_prediction(model, scaler, features)
            except Exception as exc:
                st.error(
                    "Prediction failed — the input columns could not be aligned "
                    f"to the model's expected features.\n\n**Details:** {exc}\n\n"
                    "See the *Model Info* tab for the exact feature list this "
                    "model expects."
                )
                st.stop()

        st.subheader("Prediction Result")

        if result["kind"] == "regressor":
            value = result["value"]
            label = "Predicted ADR (price per night)" if purpose == "price" \
                else "Predicted value"
            st.metric(label, f"{value:,.2f}")
            st.caption(
                "Regression output. Note: for an accurate figure the model uses "
                "all ~220 training features; fields you didn't set default to 0."
            )
        else:
            proba = result["proba"]
            if purpose == "upsell":
                pos, neg = "Likely to upsell", "Unlikely to upsell"
            else:
                pos, neg = "High risk of cancellation", "Likely to check in"
            m1, m2 = st.columns([2, 1])
            with m1:
                if proba > 0.50:
                    st.error(f"⚠️ **{pos}** — probability **{proba * 100:.1f}%**")
                else:
                    st.success(f"✅ **{neg}** — probability **{proba * 100:.1f}%**")
                st.progress(proba)
            with m2:
                st.metric("Positive-class probability", f"{proba * 100:.1f}%")

        with st.expander("Show inputs sent to the model"):
            st.json(raw_inputs)


# ---- Data & EDA tab ------------------------------------------------------- #
with data_tab:
    st.subheader("Dataset")
    if dataset is None:
        st.info(
            "No dataset found. Add **`hotel_bookings.csv`** to the repo root to "
            "enable exploratory data analysis here (optional)."
        )
    else:
        st.write(f"**{len(dataset):,}** rows × **{dataset.shape[1]}** columns")
        st.dataframe(dataset.head(200), use_container_width=True)

        if "is_canceled" in dataset.columns:
            rate = dataset["is_canceled"].mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("Overall cancellation rate", f"{rate * 100:.1f}%")
            if "hotel" in dataset.columns:
                by_hotel = dataset.groupby("hotel")["is_canceled"].mean()
                st.markdown("**Cancellation rate by hotel type**")
                st.bar_chart(by_hotel)
            if "lead_time" in dataset.columns:
                st.markdown("**Lead time distribution**")
                st.bar_chart(
                    np.histogram(dataset["lead_time"].dropna(), bins=40)[0]
                )

        with st.expander("Column dtypes & summary statistics"):
            st.write(dataset.dtypes.astype(str))
            st.dataframe(dataset.describe(include="all").T, use_container_width=True)


# ---- Model Info tab ------------------------------------------------------- #
with models_tab:
    st.subheader("Loaded artifacts")
    if not artifacts["objects"]:
        st.info(
            "No artifacts loaded yet. Place `.pkl` / `.joblib` files (or a "
            "`.zip` of them) in **`hotel_ai_models/`**."
        )
    else:
        for name, obj in artifacts["objects"].items():
            kind = (
                "model" if name in artifacts["models"]
                else "scaler/transformer" if name in artifacts["scalers"]
                else "other"
            )
            with st.expander(f"`{name}` — {type(obj).__name__} ({kind})"):
                if hasattr(obj, "feature_names_in_"):
                    cols = list(obj.feature_names_in_)
                    st.write(f"**Expects {len(cols)} features:**")
                    st.code(", ".join(map(str, cols)))
                elif hasattr(obj, "n_features_in_"):
                    st.write(f"**Expects {obj.n_features_in_} features** "
                             "(names not stored).")
                params = getattr(obj, "get_params", lambda: {})()
                if params:
                    st.write("**Parameters:**")
                    st.json({k: str(v) for k, v in params.items()})

    if artifacts["errors"]:
        st.error("Some files failed to load:")
        for err in artifacts["errors"]:
            st.write(f"- {err}")

    st.divider()
    st.caption(
        "Tip: models trained on a pandas DataFrame carry `feature_names_in_`, "
        "which this app uses to align the input row exactly. If your model was "
        "trained on a NumPy array, retrain/export on a DataFrame so column order "
        "is preserved."
    )


# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.sidebar.divider()
st.sidebar.caption(
    f"Models dir: `{MODELS_DIR.name}/` · "
    f"{len(artifacts['models'])} model(s), {len(artifacts['scalers'])} scaler(s)"
)
