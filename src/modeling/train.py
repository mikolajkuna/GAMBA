# src/modeling/train.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.utils import compute_sample_weight
from pygam import LinearGAM, s, f, te
import matplotlib.pyplot as plt
from src.config import (
    GENDER_MAP, FEATURES, GAM_TERMS, INTERACTIONS, TARGET, 
    JOB_MAP, MIN_WAGE_PLN, DISTANCE_THRESHOLD_KM, 
    CLASS_WEIGHT_MODE, MIN_GROUP_SIZE
)

# Importy dla Bayesian GAM
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("Warning: PyMC not available. Install with: pip install pymc arviz")


# =====================================================
# === LOAD & PREPROCESS DATA =========================
# =====================================================
def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        sep = ";" if ";" in f.readline() else ","
    return pd.read_csv(path, sep=sep)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"] = df["gender"].map(GENDER_MAP)

    numeric_cols = FEATURES

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # distance_from_home jako binarna 0/1
    df["distance_from_home"] = (df["distance_from_home"] >= DISTANCE_THRESHOLD_KM).astype(int)

    # Filtrujemy minimalną płacę i brak ujemnych wartości
    df = df[(df[TARGET] >= MIN_WAGE_PLN) & (df[numeric_cols].ge(0).all(axis=1))]
    return df.dropna()


# =====================================================
# === BUILD GAM TERMS ================================
# =====================================================
def build_gam_terms():
    """Buduje terminy GAM z konfiguracji"""
    terms = None

    # Dodajemy terminy z configu (GAM_TERMS)
    for feature, spec in GAM_TERMS.items():
        if spec["type"] == "s":
            term = s(FEATURES.index(feature), constraints=spec.get("constraint"))
        elif spec["type"] == "f":
            term = f(FEATURES.index(feature))
        elif spec["type"] == "te":
            term = te(FEATURES.index(spec["features"][0]), FEATURES.index(spec["features"][1]), lam=spec["lam"])
        
        if terms is None:
            terms = term
        else:
            terms = terms + term

    # Dodajemy terminy interakcji z INTERACTIONS
    for interaction in INTERACTIONS:
        feature_idx_1 = FEATURES.index(interaction["features"][0])
        feature_idx_2 = FEATURES.index(interaction["features"][1])
        term = te(feature_idx_1, feature_idx_2, lam=interaction["lam"])
        terms = terms + term
    
    return terms


# =====================================================
# === EVALUATE AND PRINT RESULTS =====================
# =====================================================
def evaluate_and_print(model, X, y, model_name="GAM", is_bayesian=False, trace=None):
    """Oblicza i wyświetla wyniki dla danego modelu"""
    
    # Performance
    if is_bayesian and trace is not None:
        # Dla modelu bayesowskiego używamy średniej z posterior
        preds = model['predictions']
    else:
        preds = model.predict(X)
    
    mae = mean_absolute_error(y, preds)
    cv = np.std(y - preds) / np.mean(y) * 100

    # Counterfactual Gender Pay Gap
    if is_bayesian and trace is not None:
        # Dla Bayesian GAM używamy zapisanych predykcji
        gap_adjusted = model['gap_adjusted']
    else:
        X_cf_m = X.copy()
        X_cf_f = X.copy()
        X_cf_m[:, FEATURES.index("gender")] = 1
        X_cf_f[:, FEATURES.index("gender")] = 0

        pred_m = model.predict(X_cf_m)
        pred_f = model.predict(X_cf_f)
        gap_adjusted = pred_m - pred_f

    # Print Results
    print(f"\n{'='*60}")
    print(f"--- {model_name} ---")
    print(f"{'='*60}")
    
    if not is_bayesian:
        print("GCV:", model.statistics_["GCV"])
        print("AIC:", model.statistics_["AIC"])
    
    print(f"\n--- PERFORMANCE ({model_name}) ---")
    print(f"MAE: {mae:,.0f} PLN")
    print(f"CV: {cv:.2f}%")

    print(f"\n--- GENDER PAY GAP ({model_name}) ---")
    if is_bayesian and 'gap_hdi' in model:
        print(f"Adjusted Mean gap (counterfactual):   {gap_adjusted.mean():,.0f} PLN")
        print(f"Adjusted Median gap (counterfactual): {np.median(gap_adjusted):,.0f} PLN")
        print(f"94% HDI: [{model['gap_hdi'][0]:,.0f}, {model['gap_hdi'][1]:,.0f}] PLN")
        if 'gap_prob_positive' in model:
            print(f"P(gap > 0): {model['gap_prob_positive']:.1%}")
    else:
        print(f"Adjusted Mean gap (counterfactual):   {gap_adjusted.mean():,.0f} PLN")
        print(f"Adjusted Median gap (counterfactual): {np.median(gap_adjusted):,.0f} PLN")

    print(f"\n--- GAP BY JOB LEVEL ({model_name} - Adjusted/Counterfactual) ---")
    for lvl in sorted(np.unique(X[:, FEATURES.index("job_level")])):
        mask = X[:, FEATURES.index("job_level")] == lvl
        if mask.sum() < MIN_GROUP_SIZE:
            continue
        print(
            f"Job level {JOB_MAP[int(lvl)]} | "
            f"mean gap: {gap_adjusted[mask].mean():,.0f} PLN | "
            f"n={mask.sum()}"
        )
    
    return gap_adjusted, mae, cv


# =====================================================
# === PLOT GAP BY JOB LEVEL ==========================
# =====================================================
def plot_gap_by_job_level(X, gap_adjusted, model_name="GAM", gap_hdi=None):
    """Tworzy wykres słupkowy luki płacowej według poziomu stanowiska"""
    job_levels = sorted(np.unique(X[:, FEATURES.index("job_level")]))
    mean_gaps = [gap_adjusted[X[:, FEATURES.index("job_level")] == lvl].mean() for lvl in job_levels]

    plt.figure(figsize=(7, 5))
    plt.bar([JOB_MAP[int(lvl)] for lvl in job_levels], mean_gaps, color='skyblue')
    plt.xlabel("Job Level")
    plt.ylabel("Mean Gender Pay Gap (PLN)")
    plt.title(f"{model_name}: Mean Gender Pay Gap by Job Level")
    plt.axhline(0, color='red', linestyle='--')
    
    # Dodaj HDI jeśli dostępne (dla Bayesian GAM)
    if gap_hdi is not None:
        plt.axhline(gap_hdi[0], color='red', linestyle=':', alpha=0.5, label='94% HDI')
        plt.axhline(gap_hdi[1], color='red', linestyle=':', alpha=0.5)
        plt.legend()
    
    plt.tight_layout()
    plt.show()


# =====================================================
# === TRAIN LINEAR GAM (Standard GCV) ================
# =====================================================
def train_linear_gam(synthetic_path: str):
    """Trenuje Linear GAM ze standardowym GCV (default)"""
    
    # Wczytanie i przetworzenie danych
    synthetic_df = preprocess(load_csv(synthetic_path))

    X = synthetic_df[FEATURES].astype(np.float32).to_numpy()
    y = synthetic_df[TARGET].astype(np.float32).to_numpy()

    # Sample weights
    weights = compute_sample_weight(
        class_weight=CLASS_WEIGHT_MODE,
        y=X[:, FEATURES.index("gender")]
    )

    # Build model
    terms = build_gam_terms()
    gam = LinearGAM(terms)

    # Train with default GCV
    gam.fit(X, y, weights=weights)

    # Evaluate and print
    gap_adjusted, mae, cv = evaluate_and_print(gam, X, y, "LINEAR GAM (default GCV)")
    
    # Plot
    plot_gap_by_job_level(X, gap_adjusted, "LINEAR GAM (default GCV)")

    return gam, gap_adjusted


# =====================================================
# === TRAIN GAM WITH GRIDSEARCH (GCV) ================
# =====================================================
def train_gridsearch_gam(synthetic_path: str):
    """
    Trenuje GAM z grid search po parametrze lambda (regularyzacja).
    Używa GCV do wyboru najlepszego modelu.
    """
    
    # Wczytanie i przetworzenie danych
    synthetic_df = preprocess(load_csv(synthetic_path))

    X = synthetic_df[FEATURES].astype(np.float32).to_numpy()
    y = synthetic_df[TARGET].astype(np.float32).to_numpy()

    # Sample weights
    weights = compute_sample_weight(
        class_weight=CLASS_WEIGHT_MODE,
        y=X[:, FEATURES.index("gender")]
    )

    # Build model
    terms = build_gam_terms()
    gam_gs = LinearGAM(terms, fit_intercept=True)

    # Grid search z GCV
    print("\nPerforming grid search with GCV objective...")
    lam = np.logspace(-3, 3, 11)  # Zakres lambda do przetestowania
    gam_gs.gridsearch(X, y, weights=weights, lam=lam, progress=True)

    # Evaluate and print
    gap_adjusted, mae, cv = evaluate_and_print(gam_gs, X, y, "GAM with GridSearch (GCV)")
    
    # Plot
    plot_gap_by_job_level(X, gap_adjusted, "GAM with GridSearch (GCV)")

    return gam_gs, gap_adjusted


# =====================================================
# === TRAIN AIC GAM ==================================
# =====================================================
def train_aic_gam(synthetic_path: str):
    """
    Trenuje GAM z AIC (Akaike Information Criterion).
    AIC penalizuje złożoność modelu bardziej niż GCV.
    """
    
    # Wczytanie i przetworzenie danych
    synthetic_df = preprocess(load_csv(synthetic_path))

    X = synthetic_df[FEATURES].astype(np.float32).to_numpy()
    y = synthetic_df[TARGET].astype(np.float32).to_numpy()

    # Sample weights
    weights = compute_sample_weight(
        class_weight=CLASS_WEIGHT_MODE,
        y=X[:, FEATURES.index("gender")]
    )

    # Build model
    terms = build_gam_terms()
    gam_aic = LinearGAM(terms, fit_intercept=True)

    # Grid search z AIC
    print("\nPerforming grid search with AIC objective...")
    lam = np.logspace(-3, 3, 11)
    gam_aic.gridsearch(X, y, weights=weights, lam=lam, objective='AIC', progress=True)

    # Evaluate and print
    gap_adjusted, mae, cv = evaluate_and_print(gam_aic, X, y, "GAM with GridSearch (AIC)")
    
    # Plot
    plot_gap_by_job_level(X, gap_adjusted, "GAM with GridSearch (AIC)")

    return gam_aic, gap_adjusted


# =====================================================
# === BAYESIAN GAM WITH PYMC =========================
# =====================================================
def train_bayesian_gam_pymc(synthetic_path: str, n_samples=2000, n_tune=1000, n_chains=2):
    """
    Trenuje prawdziwy Bayesian GAM używając PyMC.
    
    Parameters:
    -----------
    synthetic_path : str
        Ścieżka do danych
    n_samples : int
        Liczba próbek z posterior (po tuning)
    n_tune : int
        Liczba próbek do tuning (wyrzucane)
    n_chains : int
        Liczba niezależnych łańcuchów MCMC
    """
    
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC is not installed. Install with: pip install pymc arviz")
    
    print("\n" + "="*60)
    print("Building Bayesian GAM with PyMC...")
    print("="*60)
    
    # Wczytanie i przetworzenie danych
    synthetic_df = preprocess(load_csv(synthetic_path))
    X = synthetic_df[FEATURES].astype(np.float32).to_numpy()
    y = synthetic_df[TARGET].astype(np.float32).to_numpy()
    
    # Standaryzacja zmiennych ciągłych dla lepszej zbieżności
    continuous_features = ["age", "experience_years", "absence", "child"]
    X_scaled = X.copy()
    means = {}
    stds = {}
    
    for feat in continuous_features:
        idx = FEATURES.index(feat)
        means[feat] = X[:, idx].mean()
        stds[feat] = X[:, idx].std()
        X_scaled[:, idx] = (X[:, idx] - means[feat]) / stds[feat]
    
    # Indeksy cech
    age_idx = FEATURES.index("age")
    gender_idx = FEATURES.index("gender")
    edu_idx = FEATURES.index("education_level")
    job_idx = FEATURES.index("job_level")
    exp_idx = FEATURES.index("experience_years")
    dist_idx = FEATURES.index("distance_from_home")
    abs_idx = FEATURES.index("absence")
    child_idx = FEATURES.index("child")
    
    # Model PyMC
    with pm.Model() as model:
        # Priors dla współczynników liniowych
        intercept = pm.Normal("intercept", mu=5000, sigma=2000)
        
        # Continuous features - smooth effects (uproszczone)
        beta_age = pm.Normal("beta_age", mu=0, sigma=500)
        beta_exp = pm.Normal("beta_exp", mu=0, sigma=500)
        beta_abs = pm.Normal("beta_abs", mu=0, sigma=300)
        beta_child = pm.Normal("beta_child", mu=0, sigma=300)
        
        # Categorical features
        beta_gender = pm.Normal("beta_gender", mu=0, sigma=1000)
        beta_edu = pm.Normal("beta_edu", mu=0, sigma=500, shape=4)
        beta_job = pm.Normal("beta_job", mu=0, sigma=1000, shape=4)
        beta_dist = pm.Normal("beta_dist", mu=0, sigma=300)
        
        # Interactions
        beta_gender_job = pm.Normal("beta_gender_job", mu=0, sigma=500, shape=4)
        beta_gender_child = pm.Normal("beta_gender_child", mu=0, sigma=500)  # NEW: gender × child interaction
        
        # Linear predictor
        mu = (intercept + 
              beta_age * X_scaled[:, age_idx] +
              beta_exp * X_scaled[:, exp_idx] +
              beta_abs * X_scaled[:, abs_idx] +
              beta_child * X_scaled[:, child_idx] +
              beta_gender * X[:, gender_idx] +
              beta_edu[X[:, edu_idx].astype(int) - 1] +
              beta_job[X[:, job_idx].astype(int) - 1] +
              beta_dist * X[:, dist_idx] +
              beta_gender_job[X[:, job_idx].astype(int) - 1] * X[:, gender_idx] +
              beta_gender_child * X_scaled[:, child_idx] * X[:, gender_idx])  # NEW: add interaction term
        
        # Likelihood
        sigma = pm.HalfNormal("sigma", sigma=1000)
        likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        
        # Sampling
        print(f"\nSampling {n_samples} samples with {n_tune} tuning steps across {n_chains} chains...")
        print("This may take several minutes...")
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            return_inferencedata=True,
            progressbar=True,
            target_accept=0.9
        )
    
    print("\nSampling complete!")
    
    # Posterior predictive checks
    print("\nComputing posterior predictions...")
    with model:
        posterior_pred = pm.sample_posterior_predictive(trace, progressbar=True)
    
    # Wyciągnij średnie predykcje
    preds_posterior = posterior_pred.posterior_predictive['y'].mean(dim=['chain', 'draw']).values
    
    # =====================================================
    # === COUNTERFACTUAL ANALYSIS (POPRAWIONE) ===========
    # =====================================================
    print("\nComputing counterfactual gender gap...")
    X_cf_m = X_scaled.copy()
    X_cf_f = X_scaled.copy()
    X_cf_m[:, gender_idx] = 1
    X_cf_f[:, gender_idx] = 0
    
    # Wyciągnij próbki z posterior (flatten chains)
    intercept_samples = trace.posterior['intercept'].values.reshape(-1)
    beta_age_samples = trace.posterior['beta_age'].values.reshape(-1)
    beta_exp_samples = trace.posterior['beta_exp'].values.reshape(-1)
    beta_abs_samples = trace.posterior['beta_abs'].values.reshape(-1)
    beta_child_samples = trace.posterior['beta_child'].values.reshape(-1)
    beta_gender_samples = trace.posterior['beta_gender'].values.reshape(-1)
    beta_edu_samples = trace.posterior['beta_edu'].values.reshape(-1, 4)
    beta_job_samples = trace.posterior['beta_job'].values.reshape(-1, 4)
    beta_dist_samples = trace.posterior['beta_dist'].values.reshape(-1)
    beta_gender_job_samples = trace.posterior['beta_gender_job'].values.reshape(-1, 4)
    beta_gender_child_samples = trace.posterior['beta_gender_child'].values.reshape(-1)  # NEW
    
    n_mcmc_samples = len(intercept_samples)
    n_obs = len(X)
    
    # Compute predictions for each MCMC sample
    pred_m_all = np.zeros((n_mcmc_samples, n_obs))
    pred_f_all = np.zeros((n_mcmc_samples, n_obs))
    
    print("Computing posterior predictions for counterfactuals...")
    for i in range(n_mcmc_samples):
        # Male predictions
        pred_m_all[i, :] = (
            intercept_samples[i] +
            beta_age_samples[i] * X_cf_m[:, age_idx] +
            beta_exp_samples[i] * X_cf_m[:, exp_idx] +
            beta_abs_samples[i] * X_cf_m[:, abs_idx] +
            beta_child_samples[i] * X_cf_m[:, child_idx] +
            beta_gender_samples[i] * 1 +
            beta_edu_samples[i, (X_cf_m[:, edu_idx].astype(int) - 1)] +
            beta_job_samples[i, (X_cf_m[:, job_idx].astype(int) - 1)] +
            beta_dist_samples[i] * X_cf_m[:, dist_idx] +
            beta_gender_job_samples[i, (X_cf_m[:, job_idx].astype(int) - 1)] * 1 +
            beta_gender_child_samples[i] * X_cf_m[:, child_idx] * 1  # NEW
        )
        
        # Female predictions
        pred_f_all[i, :] = (
            intercept_samples[i] +
            beta_age_samples[i] * X_cf_f[:, age_idx] +
            beta_exp_samples[i] * X_cf_f[:, exp_idx] +
            beta_abs_samples[i] * X_cf_f[:, abs_idx] +
            beta_child_samples[i] * X_cf_f[:, child_idx] +
            beta_gender_samples[i] * 0 +
            beta_edu_samples[i, (X_cf_f[:, edu_idx].astype(int) - 1)] +
            beta_job_samples[i, (X_cf_f[:, job_idx].astype(int) - 1)] +
            beta_dist_samples[i] * X_cf_f[:, dist_idx] +
            beta_gender_job_samples[i, (X_cf_f[:, job_idx].astype(int) - 1)] * 0 +
            beta_gender_child_samples[i] * X_cf_f[:, child_idx] * 0  # NEW: no interaction for females
        )
    
    # Gap dla każdej obserwacji (averaged over MCMC samples)
    gap_adjusted = pred_m_all.mean(axis=0) - pred_f_all.mean(axis=0)
    
    # Gap samples (dla HDI) - średnia luka w populacji dla każdej próbki MCMC
    gap_mean_samples = (pred_m_all - pred_f_all).mean(axis=1)
    
    # Compute HDI poprawnie
    gap_hdi = az.hdi(gap_mean_samples, hdi_prob=0.94)
    gap_prob_positive = (gap_mean_samples > 0).mean()
    gap_prob_over_50 = (gap_mean_samples > 50).mean()
    
    # =====================================================
    # === DIAGNOSTICS ====================================
    # =====================================================
    print("\n" + "="*60)
    print("MCMC Diagnostics:")
    print("="*60)
    print(az.summary(trace, var_names=["intercept", "beta_gender", "beta_gender_child", "sigma"], hdi_prob=0.94))
    
    print("\n" + "="*60)
    print("Gender Pay Gap - Bayesian Posterior:")
    print("="*60)
    print(f"Mean gap: {gap_mean_samples.mean():,.0f} PLN")
    print(f"Median gap: {np.median(gap_mean_samples):,.0f} PLN")
    print(f"Std dev: {gap_mean_samples.std():,.0f} PLN")
    print(f"94% HDI: [{gap_hdi[0]:,.0f}, {gap_hdi[1]:,.0f}] PLN")
    print(f"\nProbability gap > 0: {gap_prob_positive:.1%}")
    print(f"Probability gap > 50 PLN: {gap_prob_over_50:.1%}")
    
    # =====================================================
    # === MOTHERHOOD PENALTY ANALYSIS ====================
    # =====================================================
    print("\n" + "="*60)
    print("Motherhood Penalty Analysis (gender × child):")
    print("="*60)
    
    # Extract beta_gender_child posterior
    beta_gender_child_posterior = trace.posterior['beta_gender_child'].values.reshape(-1)
    beta_gender_child_hdi = az.hdi(beta_gender_child_posterior, hdi_prob=0.94)
    
    print(f"beta_gender_child (interaction effect):")
    print(f"  Mean: {beta_gender_child_posterior.mean():,.0f} PLN")
    print(f"  Median: {np.median(beta_gender_child_posterior):,.0f} PLN")
    print(f"  94% HDI: [{beta_gender_child_hdi[0]:,.0f}, {beta_gender_child_hdi[1]:,.0f}] PLN")
    print(f"  P(effect > 0): {(beta_gender_child_posterior > 0).mean():.1%}")
    print(f"  P(effect < 0): {(beta_gender_child_posterior < 0).mean():.1%}")
    
    print("\nInterpretation:")
    if beta_gender_child_posterior.mean() > 0:
        print(f"  • Men with children earn MORE (+{beta_gender_child_posterior.mean():.0f} PLN per child)")
        print(f"  • This suggests a 'fatherhood bonus'")
    else:
        print(f"  • Men with children earn LESS ({beta_gender_child_posterior.mean():.0f} PLN per child)")
        print(f"  • This is unexpected - fatherhood typically increases earnings")
    
    # Compute gap stratified by children
    print("\n" + "="*60)
    print("Gender Pay Gap by Number of Children:")
    print("="*60)
    
    unique_children = sorted(np.unique(X[:, child_idx]))
    for n_child in unique_children:
        mask = X[:, child_idx] == n_child
        if mask.sum() < MIN_GROUP_SIZE:
            continue
        gap_by_child = gap_adjusted[mask]
        print(f"  {int(n_child)} children: mean gap = {gap_by_child.mean():,.0f} PLN (n={mask.sum()})")
    
    
    # Zwracamy słownik z wynikami
    bayesian_model = {
        'trace': trace,
        'predictions': preds_posterior,
        'gap_adjusted': gap_adjusted,
        'gap_hdi': (gap_hdi[0], gap_hdi[1]),
        'gap_mean_samples': gap_mean_samples,
        'gap_prob_positive': gap_prob_positive,
        'model': model
    }
    
    # Evaluate and print
    gap_adjusted, mae, cv = evaluate_and_print(
        bayesian_model, X, y, "BAYESIAN GAM (PyMC)", 
        is_bayesian=True, trace=trace
    )
    
    # Plot
    plot_gap_by_job_level(X, gap_adjusted, "BAYESIAN GAM (PyMC)", gap_hdi=bayesian_model['gap_hdi'])
    
    return bayesian_model, gap_adjusted


# =====================================================
# === TRAIN ALL MODELS ===============================
# =====================================================
def train_all_gams(synthetic_path: str, include_bayesian=False):
    """
    Trenuje wszystkie warianty modeli i porównuje wyniki
    
    Parameters:
    -----------
    include_bayesian : bool
        Czy włączyć Bayesian GAM (może być wolny)
    """
    
    print("\n" + "="*60)
    print("TRAINING LINEAR GAM (default GCV)")
    print("="*60)
    default_gam, default_gap = train_linear_gam(synthetic_path)
    
    print("\n\n" + "="*60)
    print("TRAINING GAM WITH GRIDSEARCH (GCV)")
    print("="*60)
    gcv_gam, gcv_gap = train_gridsearch_gam(synthetic_path)
    
    print("\n\n" + "="*60)
    print("TRAINING GAM WITH GRIDSEARCH (AIC)")
    print("="*60)
    aic_gam, aic_gap = train_aic_gam(synthetic_path)
    
    results = {
        'default': {'model': default_gam, 'gap': default_gap},
        'gridsearch_gcv': {'model': gcv_gam, 'gap': gcv_gap},
        'gridsearch_aic': {'model': aic_gam, 'gap': aic_gap}
    }
    
    if include_bayesian and PYMC_AVAILABLE:
        print("\n\n" + "="*60)
        print("TRAINING BAYESIAN GAM (PyMC)")
        print("="*60)
        bayes_model, bayes_gap = train_bayesian_gam_pymc(synthetic_path)
        results['bayesian'] = {'model': bayes_model, 'gap': bayes_gap}
    
    # Porównanie
    print("\n\n" + "="*60)
    print("COMPARISON OF ALL MODELS")
    print("="*60)
    print(f"Default GCV      - Mean Gap: {default_gap.mean():,.0f} PLN")
    print(f"GridSearch (GCV) - Mean Gap: {gcv_gap.mean():,.0f} PLN")
    print(f"GridSearch (AIC) - Mean Gap: {aic_gap.mean():,.0f} PLN")
    if include_bayesian and 'bayesian' in results:
        print(f"Bayesian (PyMC)  - Mean Gap: {bayes_gap.mean():,.0f} PLN")
        print(f"                   94% HDI: [{results['bayesian']['model']['gap_hdi'][0]:,.0f}, {results['bayesian']['model']['gap_hdi'][1]:,.0f}] PLN")
        print(f"                   P(gap > 0): {results['bayesian']['model']['gap_prob_positive']:.1%}")
    
    return results


# =====================================================
# === BACKWARD COMPATIBILITY =========================
# =====================================================
def train_gamba(synthetic_path: str):
    """Alias dla train_linear_gam dla zachowania wstecznej kompatybilności"""
    return train_linear_gam(synthetic_path)

train_both_gams = train_all_gams
