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
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except Exception:  # pragma: no cover - joblib is in requirements.txt
    joblib = None

# Load a local .env (if present) so GEMINI_API_KEY and friends land in the
# environment without any UI. Safe no-op when python-dotenv isn't installed or
# there is no .env file (e.g. on Streamlit Cloud, where Secrets are used).
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass


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
TEST_DATA_CANDIDATES = [
    APP_DIR / "app_test_data.csv",
    APP_DIR / "test_data.csv",
    APP_DIR / "data" / "app_test_data.csv",
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


@st.cache_data(show_spinner=False)
def load_test_data() -> pd.DataFrame | None:
    """Load the held-out test rows the user can pick from and run predictions on.

    Rows are raw (un-encoded) booking records — the same feature space the
    sidebar collects — so any row can flow straight through ``build_feature_row``.
    """
    for candidate in TEST_DATA_CANDIDATES:
        if candidate.exists():
            try:
                df = pd.read_csv(candidate)
                # NaNs in agent/children/country etc. break encoding/scaling.
                num = df.select_dtypes(include="number").columns
                df[num] = df[num].fillna(0)
                obj = df.select_dtypes(exclude="number").columns
                df[obj] = df[obj].fillna("Undefined")
                return df
            except Exception:
                return None
    return None


def row_to_raw(row: pd.Series) -> dict:
    """Convert a raw test-data row into the input dict ``build_feature_row`` uses.

    The test CSV's columns already match the model feature space, so this is
    largely a passthrough; it just coerces obvious integer-like fields.
    """
    raw = row.to_dict()
    for key in ("is_repeated_guest", "is_canceled", "adults", "children",
                "babies", "total_of_special_requests", "booking_changes",
                "previous_cancellations", "previous_bookings_not_canceled",
                "required_car_parking_spaces", "days_in_waiting_list",
                "arrival_date_year", "arrival_date_week_number",
                "arrival_date_day_of_month"):
        if key in raw and pd.notna(raw[key]):
            try:
                raw[key] = int(raw[key])
            except (TypeError, ValueError):
                pass
    return raw


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


def is_tree_based(model) -> bool:
    """True for scale-invariant tree/ensemble models (which must NOT be scaled).

    The tree models here were trained on raw, unscaled features; applying the
    StandardScaler to them badly degrades accuracy. Only scale-sensitive models
    (LogisticRegression, SVM, KNN, …) should receive the scaler.
    """
    name = type(model).__name__.lower()
    return any(k in name for k in (
        "forest", "tree", "boost", "xgb", "lgbm", "lightgbm", "catboost",
        "bagging",
    ))


def pick_scaler(model, scalers: dict, model_name: str = ""):
    """Choose the scaler that matches this model, or ``None``.

    Different model families here have different feature spaces (cancellation vs
    upsell vs pricing) *and* different scaling needs.

    0. Tree/ensemble models are scale-invariant and were trained unscaled → never
       scale them (see ``is_tree_based``).
    1. If the model carries ``feature_names_in_``, pair it with the scaler whose
       column set matches exactly — otherwise scale nothing.
    2. If the model has *no* feature names (e.g. LogisticRegression trained on a
       scaled array), fall back to matching by feature count, disambiguating by
       name family (``upsell`` vs cancellation ``base``).
    """
    if not scalers or is_tree_based(model):
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

predict_tab, data_tab, models_tab, advisor_tab = st.tabs(
    ["🔮 Predict", "📊 Data & EDA", "🧠 Model Info", "🤖 AI Advisor"]
)


def _model_purpose(name: str) -> str:
    """Best-effort human label for what a model predicts, from its filename."""
    low = name.lower()
    if "pricing" in low or "regress" in low or "adr" in low:
        return "price"          # regression → predicted ADR
    if "upsell" in low:
        return "upsell"         # classification → upsell likelihood
    return "cancellation"       # default → cancellation risk


def _true_target(row: pd.Series, purpose: str):
    """Extract the ground-truth value from a test row for the given purpose."""
    if purpose == "price" and "adr" in row:
        return float(row["adr"])
    if purpose == "upsell" and "total_of_special_requests" in row:
        return int(row["total_of_special_requests"]) > 0
    if "is_canceled" in row:
        return int(row["is_canceled"])
    return None


def _find_model(artifacts: dict, purpose: str, prefer: str = "") -> str | None:
    """Pick a loaded model of a given purpose, preferring a name substring."""
    names = [n for n in artifacts["models"] if _model_purpose(n) == purpose]
    if not names:
        return None
    for n in names:
        if prefer and prefer in n:
            return n
    return names[0]


def summarize_forecast(artifacts: dict, test_df: pd.DataFrame,
                       n: int = 50, seed: int = 42) -> str:
    """Run the ML models over a sample of bookings and describe the week.

    Produces the plain-text forecast the CrewAI agents reason over: predicted
    cancellation rate, expected check-ins, high-risk exposure, and the pricing
    model's average predicted ADR / projected revenue.
    """
    sample = test_df.sample(min(n, len(test_df)), random_state=seed)

    cx_name = _find_model(artifacts, "cancellation", prefer="optimized_rf")
    price_name = _find_model(artifacts, "price")

    lines = [f"Sample size: {len(sample)} upcoming bookings."]

    if cx_name:
        model = artifacts["models"][cx_name]
        scaler = pick_scaler(model, artifacts["scalers"], cx_name)
        exp = expected_features(scaler, model)
        probs = []
        for _, row in sample.iterrows():
            feats = build_feature_row(row_to_raw(row), exp)
            res = run_prediction(model, scaler, feats)
            probs.append(res.get("proba", float(res.get("label", 0))))
        probs = np.array(probs)
        cancel_rate = float((probs > 0.5).mean())
        high_risk = int((probs > 0.7).sum())
        lines += [
            f"Cancellation model: {cx_name}",
            f"Predicted cancellation rate: {cancel_rate * 100:.1f}%",
            f"Expected check-ins: {int(round((1 - cancel_rate) * len(sample)))} "
            f"of {len(sample)}",
            f"High-risk bookings (>70% cancel prob): {high_risk}",
        ]

    if price_name:
        model = artifacts["models"][price_name]
        scaler = pick_scaler(model, artifacts["scalers"], price_name)
        exp = expected_features(scaler, model)
        adrs = []
        for _, row in sample.iterrows():
            feats = build_feature_row(row_to_raw(row), exp)
            adrs.append(run_prediction(model, scaler, feats)["value"])
        adrs = np.array(adrs)
        lines += [
            f"Pricing model: {price_name}",
            f"Average predicted ADR: {adrs.mean():.2f}",
            f"ADR range: {adrs.min():.2f} - {adrs.max():.2f}",
            f"Projected room revenue (ADR sum): {adrs.sum():,.0f}",
        ]

    return "\n".join(lines)


# ---- Predict tab ---------------------------------------------------------- #
with predict_tab:
    model_names = list(artifacts["models"].keys())
    test_df = load_test_data()

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

    # --- Input source: a test-data row, or manual entry ---------------------
    default_source = "🎲 Test data row" if test_df is not None else "✍️ Manual input"
    source = st.radio(
        "Input source", ["🎲 Test data row", "✍️ Manual input"],
        horizontal=True,
        index=0 if default_source.startswith("🎲") else 1,
    )

    selected_row = None
    if source.startswith("🎲"):
        if test_df is None:
            st.warning(
                "No test data found. Add **`app_test_data.csv`** to the repo "
                "root (raw booking rows) to enable this. Falling back to manual "
                "input below."
            )
            raw_inputs = collect_raw_inputs(country_codes(artifacts))
        else:
            n_rows = len(test_df)
            st.session_state.setdefault("test_row_idx", random.randint(0, n_rows - 1))
            c1, c2 = st.columns([1, 2])
            with c1:
                if st.button("🎲 Pick a random row", use_container_width=True):
                    st.session_state["test_row_idx"] = random.randint(0, n_rows - 1)
            with c2:
                st.number_input(
                    f"…or choose a row (0–{n_rows - 1})", min_value=0,
                    max_value=n_rows - 1, step=1, key="test_row_idx",
                )
            idx = int(st.session_state["test_row_idx"])
            selected_row = test_df.iloc[idx]
            st.caption(f"Selected **row {idx}** of {n_rows:,} test bookings:")
            st.dataframe(selected_row.to_frame().T, use_container_width=True)
            raw_inputs = row_to_raw(selected_row)
    else:
        raw_inputs = collect_raw_inputs(country_codes(artifacts))

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

        # For a test row, compare the prediction against the ground truth.
        if selected_row is not None:
            truth = _true_target(selected_row, purpose)
            if truth is not None:
                st.markdown("**Actual vs. predicted** (ground truth from test data)")
                a, b = st.columns(2)
                if result["kind"] == "regressor":
                    pred_v = result["value"]
                    a.metric("Actual ADR", f"{float(truth):,.2f}")
                    b.metric("Predicted ADR", f"{pred_v:,.2f}",
                             delta=f"{pred_v - float(truth):,.2f}")
                else:
                    if purpose == "upsell":
                        actual_txt = "Upsell" if truth else "No upsell"
                        pred_txt = "Upsell" if result["label"] else "No upsell"
                    else:
                        actual_txt = "Cancelled" if truth else "Not cancelled"
                        pred_txt = "Cancelled" if result["label"] else "Not cancelled"
                    a.metric("Actual", actual_txt)
                    b.metric("Predicted", pred_txt)
                    if int(bool(truth)) == int(result["label"]):
                        st.success("✅ Prediction matches the actual outcome.")
                    else:
                        st.warning("❌ Prediction differs from the actual outcome.")

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


# ---- AI Advisor tab ------------------------------------------------------- #
with advisor_tab:
    st.subheader("🤖 AI Revenue Advisor")
    st.markdown(
        "Two **CrewAI** agents powered by **Gemini** turn this week's ML forecast "
        "into an action plan:\n"
        "1. **Senior Hotel Data Analyst** — reads the model outputs and briefs the "
        "risks & opportunities.\n"
        "2. **Director of Revenue Management** — converts that into concrete "
        "pricing & overbooking actions."
    )

    import crew_agents

    ok, err = crew_agents.crewai_available()
    if not ok:
        st.warning(
            "CrewAI isn't installed in this environment, so the crew can't run "
            "here. It's listed in `requirements.txt`, so Streamlit Cloud will "
            "install it on deploy. To run locally:\n\n"
            "```bash\npip install crewai\n```\n\n"
            f"_Import error: {err}_"
        )

    # --- API key: prefer Streamlit secrets, else a password field -----------
    secret_key = ""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    env_key = os.environ.get("GEMINI_API_KEY", "")

    key_col, model_col, n_col = st.columns([2, 1, 1])
    with key_col:
        if secret_key or env_key:
            src = "secrets" if secret_key else ".env / environment"
            st.success(f"Gemini API key found in {src} ✅")
            api_key = secret_key or env_key
        else:
            api_key = st.text_input(
                "Gemini API key", type="password",
                help="Local: put GEMINI_API_KEY in a .env file (auto-loaded, "
                     "gitignored). On Streamlit Cloud: add it under "
                     "Settings → Secrets. This field is a session-only fallback.",
            )
    with model_col:
        gemini_model = st.selectbox(
            "Gemini model",
            ["gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro"],
        )
    with n_col:
        sample_n = st.number_input("Bookings to forecast", 10, 500, 50, step=10)

    advisor_test_df = load_test_data()
    if advisor_test_df is None:
        st.info("Add `app_test_data.csv` to generate a forecast for the agents.")
    can_run = ok and bool(api_key) and advisor_test_df is not None and bool(model_names)

    if st.button("Generate AI strategy report", type="primary",
                 disabled=not can_run, use_container_width=True):
        with st.spinner("Building forecast from the ML models…"):
            context = summarize_forecast(artifacts, advisor_test_df, n=int(sample_n))
        with st.expander("Forecast handed to the agents", expanded=True):
            st.code(context)
        try:
            with st.spinner(f"Agents deliberating via {gemini_model}… "
                            "(this can take ~30-60s)"):
                report = crew_agents.run_advisor(
                    context, api_key=api_key, llm=gemini_model, verbose=False
                )
            st.markdown("### 📋 Revenue action plan")
            st.markdown(report)
        except Exception as exc:
            st.error(
                "The crew failed to produce a report.\n\n"
                f"**Details:** {exc}\n\n"
                "Common causes: an invalid/expired Gemini API key, no quota, or "
                "no network egress to Google from the deploy environment."
            )


# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.sidebar.divider()
st.sidebar.caption(
    f"Models dir: `{MODELS_DIR.name}/` · "
    f"{len(artifacts['models'])} model(s), {len(artifacts['scalers'])} scaler(s)"
)
