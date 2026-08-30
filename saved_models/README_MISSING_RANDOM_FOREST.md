# Saved Models Folder Notes

This folder was integrated from the provided `saved_models.zip`.

The following large files were intentionally not included by the user:

- `random_forest.pkl`
- `random_forest_tuned.pkl`

The project can still inspect/use the included CNN, autoencoder, XGBoost, LightGBM, Isolation Forest, scalers, and metadata files. Any script that explicitly loads `saved_models/random_forest.pkl` will still require either:

1. copying the missing RF file into this folder, or
2. switching the static Layer 1 model to the included XGBoost/LightGBM model, or
3. retraining/saving a smaller RF model.
