# Saved Models Review

The provided `saved_models.zip` was checked and integrated into this improved project package.

## Included artifacts

- CNN model and metadata:
  - `saved_models/cnn/model.keras`
  - `saved_models/cnn/meta.json`
  - `saved_models/cnn_scaler.pkl`
- Static anomaly models:
  - `saved_models/autoencoder/model.keras`
  - `saved_models/autoencoder/meta.json`
  - `saved_models/ae_scaler.pkl`
  - `saved_models/isolation_forest.pkl`
  - `saved_models/if_scaler.pkl`
- Static supervised models/scalers:
  - `saved_models/xgboost.pkl`
  - `saved_models/xgboost_tuned.pkl`
  - `saved_models/xgb_scaler.pkl`
  - `saved_models/xgb_tuned_scaler.pkl`
  - `saved_models/lightgbm.pkl`
  - `saved_models/lightgbm_tuned.pkl`
  - `saved_models/lgb_scaler.pkl`
  - `saved_models/lgb_tuned_scaler.pkl`
  - `saved_models/rf_scaler.pkl`
  - `saved_models/rf_tuned_scaler.pkl`
- LSTM autoencoder artifacts:
  - `saved_models/lstm_autoencoder.keras`
  - `saved_models/lstm_autoencoder/model.keras`
  - `saved_models/lstm_autoencoder/meta.json`
  - `saved_models/lstm_autoencoder_meta.json`
  - `saved_models/lstm_scaler.pkl`

## Compatibility findings

- The CNN metadata reports `window_size = 8`, `n_features = 10`, and the same 10 feature columns used in the improved `cnn_preprocessor.py`.
- The CNN scaler is a `RobustScaler` with 10 input features, matching the CNN metadata.
- The Layer 1 static scalers (`rf`, `xgb`, `lgb`, `ae`, `if`) use 69 features.
- The static autoencoder metadata also reports 69 features and matches the autoencoder scaler feature order.
- The LSTM autoencoder uses a different static feature schema: 55 raw memory features, not the 69 engineered-feature schema.

## Important limitation

The two large Random Forest model files are not included:

- `saved_models/random_forest.pkl`
- `saved_models/random_forest_tuned.pkl`

So any code path that directly loads the RF model will fail until those files are restored or the static model is switched to one of the included models such as XGBoost or LightGBM.

## Recommended next step

After replacing your project with this package, do not rely on the old CNN artifact for final results. The old CNN artifact was trained before the labeling fix. You should:

1. regenerate the dynamic preprocessing cache with `force_reprocess=True`,
2. retrain the CNN,
3. save the new CNN model/scaler/metadata,
4. rerun the Notepad, behavioral ransomware, and memory tests.
