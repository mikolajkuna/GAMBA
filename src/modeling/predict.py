# src/modeling/predict.py



def predict(model, X):
    return model.predict(X)


def counterfactual_gender_gap(model, X):
    X_m = X.copy()
    X_f = X.copy()

    X_m[:, 1] = 1
    X_f[:, 1] = 0

    return model.predict(X_m) - model.predict(X_f)

def predict_gam_bayes(trace, X_new, features_index):
    intercept = trace.posterior["intercept"].mean().values
    beta_age = trace.posterior["beta_age"].mean().values
    beta_exp = trace.posterior["beta_exp"].mean().values
    beta_abs = trace.posterior["beta_abs"].mean().values
    beta_child = trace.posterior["beta_child"].mean().values
    beta_gender = trace.posterior["beta_gender"].mean().values
    beta_edu = trace.posterior["beta_edu"].mean(dim=["chain", "draw"]).values
    beta_job = trace.posterior["beta_job"].mean(dim=["chain", "draw"]).values
    beta_dist = trace.posterior["beta_dist"].mean().values
    beta_gender_job = trace.posterior["beta_gender_job"].mean(dim=["chain", "draw"]).values
    beta_gender_child = trace.posterior["beta_gender_child"].mean().values

    preds = (
        intercept
        + beta_age * X_new[:, features_index["age"]]
        + beta_exp * X_new[:, features_index["experience_years"]]
        + beta_abs * X_new[:, features_index["absence"]]
        + beta_child * X_new[:, features_index["child"]]
        + beta_gender * X_new[:, features_index["gender"]]
        + beta_edu[X_new[:, features_index["education_level"]].astype(int) - 1]
        + beta_job[X_new[:, features_index["job_level"]].astype(int) - 1]
        + beta_dist * X_new[:, features_index["distance_from_home"]]
        + beta_gender_job[X_new[:, features_index["job_level"]].astype(int) - 1] * X_new[:, features_index["gender"]]
        + beta_gender_child * X_new[:, features_index["child"]] * X_new[:, features_index["gender"]]
    )
    return preds
