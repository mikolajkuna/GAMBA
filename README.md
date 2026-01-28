# GAMBA – Generalized Additive Model for Bias Analysis

This repository implements **GAMBA** (Generalized Additive Model for Bias Analysis), a framework to analyze and model salary data with counterfactual gender adjustments. It demonstrates **gender pay gap analysis**, predictive modeling using **GAMs**, and visualization of effects across features like job level and number of children.

It is built on the [cookiecutter data science template](https://github.com/drivendataorg/cookiecutter-data-science), providing a clean, reproducible structure and CI integration with **pytest** and **flake8**.

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

The full file structure is described below.

---

### How to run the code

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

## Project Organization

```
├── README.md          
├── data
│   ├── external       
│   ├── interim        
│   ├── processed      
│   └── raw            
│
├── docs               
│
├── models             
│
├── notebooks          
│
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

## Environment Setup

### Using `venv`

```bash
python3 -m venv .venv          # Create virtual environment
source .venv/bin/activate      # Activate environment
pip install -r requirements.txt  # Install all dependencies
```

## GAMBA Highlights

* **Counterfactual Gender Pay Gap Analysis**: Predict salaries if all employees were male/female.
* **Flexible GAM specification**: monotonicity constraints on age, experience, job level, absence, etc.
* **Feature-specific insights**: visualizations for job level, number of children, and combined interactions (gender × job level).
* **Weighted modeling**: balances gender distribution in training with sample weights.
* **Reproducible pipeline**: fully modular `dataset -> features -> modeling -> plots`.

---

## Additional Tips

* Keep `config.py` updated with paths, features, and GAM settings.
* Use `reports/figures` for all plots to maintain reproducibility.
* Tests live in `tests/` (pytest compatible) to ensure preprocessing and modeling correctness.

