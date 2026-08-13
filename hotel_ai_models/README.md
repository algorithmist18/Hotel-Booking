# hotel_ai_models/

Drop your exported scikit-learn artifacts here. The app auto-discovers them.

Supported files (any combination):

- `*.pkl` / `*.joblib` / `*.sav` — loaded directly with `joblib.load`.
- `*.zip` — automatically extracted in place at app startup; the `.pkl` files
  inside become available (handy for keeping large forests out of Git as loose
  files).

## Classification

Each loaded object is auto-classified:

- Anything with a `.predict` method → **model** (selectable in the Predict tab).
- Anything with `.transform` (or a name containing `scaler`) → **scaler**.

So naming files clearly helps, e.g.:

```
hotel_ai_models/
├── optimized_rf_cancellation.pkl   # the classifier
├── base_scaler.pkl                 # the fitted StandardScaler
└── ...                             # any other models are selectable too
```

## Important: feature alignment

Export your models **fitted on a pandas DataFrame**, not a raw NumPy array.
scikit-learn then stores `feature_names_in_` on the estimator/scaler, and the
app uses that to align the user's inputs to the exact one-hot columns your
`pd.get_dummies()` produced during training. Without it, the app can only align
by feature count, which is fragile.

## Version note

Unpickling can break across major scikit-learn versions. If the app fails to
load a `.pkl`, pin the scikit-learn version you trained with in
`requirements.txt`, e.g. `scikit-learn==1.5.1`.
