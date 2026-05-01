# S004 — Ransomware Detection and Mitigation System
## Phase 3: Research-Grade ML Pipeline (v2)

---

## Project Structure

```
ransomware_v2/
├── notebooks/
│   └── 03_Research_Grade_Pipeline.ipynb   ← main notebook (run this)
├── src/
│   ├── config.py                          ← all hyperparameters and paths
│   ├── train.py                           ← training orchestrator (CLI)
│   ├── evaluate.py                        ← evaluation pipeline
│   ├── hyperparameter_tuning.py           ← Optuna HPO
│   ├── utils.py                           ← shared helpers
│   ├── models/
│   │   ├── data_loader.py                 ← loads CIC-MalMem2022, engineers features
│   │   ├── random_forest_model.py         ← RF with recall-optimised threshold
│   │   ├── isolation_forest_model.py      ← IF with Benign-only training (fixed)
│   │   ├── autoencoder_model.py           ← deep autoencoder (replaces LSTM)
│   │   └── ensemble_model.py              ← XGBoost + LightGBM + soft-voting ensemble
│   └── engine/
│       └── system_event.py                ← standard output format for all models
├── behavioral_data_collection_guide.md    ← VM setup + LSTM data collection
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your dataset
# Copy MalMem2022.csv to:  data/datasets/MalMem2022.csv
# Edit CICMALMEM_PATHS in src/config.py if path differs

# 3. Run the notebook
jupyter notebook notebooks/03_Research_Grade_Pipeline.ipynb

# 4. Or train from CLI
cd src/
python train.py                        # train all models
python train.py --model rf             # train only Random Forest
python train.py --skip_existing        # skip already-saved models

# 5. Hyperparameter tuning (optional, improves metrics by 2-5%)
python hyperparameter_tuning.py --model rf  --trials 50
python hyperparameter_tuning.py --model xgb --trials 100
python train.py --use_tuned_params
```

---

## What Was Fixed (v1 → v2)

| Issue | Fix |
|-------|-----|
| RF threshold at 0.48 (optimised for F1) | Tuned for **recall ≥ 90%** on validation set |
| Isolation Forest trained on mixed classes | Now trained on **Benign-only** rows |
| IF contamination=0.05 | Fixed to **0.165** (actual ransomware ratio) |
| LSTM on static snapshots (wrong design) | Replaced with **Deep Autoencoder** |
| No feature engineering | Added **15 ratio/interaction features** → 70 total |
| Only RF as baseline | Added **XGBoost + LightGBM + Ensemble** |
| No ablation study | Added raw vs engineered feature comparison |
| No probability calibration | Added **isotonic calibration** |

---

## Expected Metrics (CIC-MalMem2022)

| Model | Recall | ROC-AUC |
|-------|--------|---------|
| Random Forest | ≥ 90% | 96–97% |
| XGBoost | 91–94% | 97% |
| LightGBM | 90–93% | 97% |
| Ensemble | 93–95% | 98% |
| Isolation Forest | 70–80% | 68–78% |
| Autoencoder | 65–80% | 70–82% |

---

## Dataset

**CIC-MalMem2022** — 59,446 memory forensics samples  
- Ransomware: 9,791 (16.5%)  
- Benign: 29,298 | Spyware: 10,870 | Trojan: 9,487  
- 55 raw Volatility features + 15 engineered = 70 total

**To improve results:** Download additional Ransomware + Benign CSV files  
from https://www.unb.ca/cic/datasets/malmem-2022.html  
and add their paths to `CICMALMEM_PATHS` in `src/config.py`.
