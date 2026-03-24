# GAMBA – Generalized Additive Model and Bayesian Analysis

This repository implements **GAMBA** (Generalized Additive Model and Bayesian Analysis), a framework for pay equity analysis using GAMs with frequentist (REML) and Bayesian (MCMC) estimation. It demonstrates **gender pay gap analysis**, predictive modeling, and visualization of interaction effects across job level and family status.

It is built on the [cookiecutter data science template](https://github.com/drivendataorg/cookiecutter-data-science), providing a clean, reproducible structure and CI integration with **pytest** and **flake8**.

This repository accompanies the paper:
> Kuna, M., Kowalczyk, M.: *Bayesian and Frequentist Approaches to Pay Equity Analysis Using Generalized Additive Models*. PP-RAI 2026.

---

## What is here?

### Code and Files Structure

The repository follows a clear modular design. Each step – from data ingestion to feature engineering, model training, prediction, and visualization – has its own module under `src/`.

* `dataset.py` – loading raw CSV data
* `features.py` – preprocessing and feature creation
* `modeling/train.py` – train the GAMBA model
* `modeling/predict.py` – generate predictions with trained models
* `plots.py` – visualizations (gender pay gap, counterfactual analysis)
* `config.py` – central place for constants, paths, feature lists, and GAM specifications

---

## Data

All datasets in this repository are **synthetic** and contain **no real individual records**. They are based on realistic distributions characteristic of Polish salary data circa 2009 (national minimum wage: 1,276 PLN).

The raw dataset is provided in `data/raw/salary_data_2009.csv`.

### Column Descriptions

| Column | Type | Values / Range | Description |
|---|---|---|---|
| `gender` | categorical | M / F | Employee gender |
| `age` | continuous | integer, years | Employee age |
| `education_level` | ordinal | 1–4 | Level of education (1 = lowest, 4 = highest) |
| `job_level` | ordinal | 1 = Junior, 2 = Mid, 3 = Senior, 4 = Manager | Hierarchical job level |
| `experience_years` | continuous | years | Total work experience |
| `distance_from_home` | binary | 0 = ≤15 km, 1 = >15 km | Whether employee lives more than 15 km from workplace |
| `absence` | discrete | count | Number of absence days |
| `child` | discrete | count (0–3+) | Number of children |
| `income` | continuous | PLN | Gross monthly salary |

### Data Generation

Income is generated according to the following structural model:

- **Age effect**: `f_age(x) = 2000·log(x−20)` — concave trajectory consistent with human capital theory
- **Experience effect**: `f_exp(x) = 1500·(1−exp(−0.3·x))` — diminishing returns after 5–7 years
- **Job level premiums**: Mid +8,000 PLN, Senior +18,000 PLN, Manager +25,000 PLN
- **Gender main effect**: +15 PLN (deliberately small for method validation)
- **Gender × job level interactions**: Junior −100, Mid +600, Senior +500, Manager +200 PLN
- **Gender × child interaction**: +200 PLN per child (fatherhood premium / motherhood penalty)
- **Noise**: Gaussian, σ = 7,000 PLN

---

## How to run the code

```bash
# Load dataset (raw)
python3 -m src.dataset

# Preprocess and create features
python3 -m src.features

# Train GAMBA model
python3 -m src.modeling.train

# Predict using trained model
python3 -m src.modeling.predict
```

Notebooks in `notebooks/` allow interactive exploration and plotting.

---

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All random seeds are fixed in the code to ensure exact replication of reported results.

---

## Project Organization

```
├── README.md
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw
│       └── salary_data_2009.csv
│
├── docs
├── models
├── notebooks
├── reports
│   └── figures
│
├── requirements.txt
├── setup.cfg
└── src
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── features.py
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py
    │   └── train.py
    └── plots.py
```

---

## GAMBA Highlights

* **Counterfactual Gender Pay Gap Analysis**: predict salaries if all employees were male/female.
* **Flexible GAM specification**: monotonicity constraints on age, experience, job level, absence, etc.
* **Bayesian modeling**: probabilistic inference and credible intervals via MCMC (PyMC, NUTS sampler).
* **Frequentist modeling**: REML-based smoothing parameter selection via pyGAM.
* **Interaction effects**: gender × job level and gender × child visualizations.
* **Reproducible pipeline**: fully modular `dataset → features → modeling → plots`.

---

## Additional Tips

* Keep `config.py` updated with paths, features, and GAM settings.
* Use `reports/figures` for all plots to maintain reproducibility.
* Tests live in `tests/` (pytest compatible) to ensure preprocessing and modeling correctness.
