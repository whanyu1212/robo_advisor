import json
import pickle

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from mlflow.exceptions import MlflowException
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RepeatedKFold, train_test_split


def create_or_get_experiment(name):
    try:
        experiment_id = mlflow.create_experiment(name)
    except MlflowException:
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is not None:
            experiment_id = experiment.experiment_id
        else:
            raise ValueError("Experiment not found.")
    return experiment_id


def log_model_and_params(model, trial, params, mean_accuracy):
    mlflow.lightgbm.log_model(model, "lightgbm_model")
    mlflow.log_params(params)
    mlflow.log_metric("mean_accuracy", mean_accuracy)
    trial.set_user_attr(key="best_booster", value=pickle.dumps(model))


def objective(X_train, y_train, trial):
    experiment_id = create_or_get_experiment("lightgbm-optuna")

    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        params = {
            "objective": "multiclass",
            "boosting_type": "gbdt",
            "num_class": 7,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        lgbm_cl = LGBMClassifier(**params)

        rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
        accuracy_scores = []
        logloss_scores = []

        X_values = X_train.values
        y_values = y_train.values
        for train_index, valid_index in rkf.split(X_train):
            X_train_sub, y_train_sub = X_values[train_index], y_values[train_index]
            X_valid_sub, y_valid_sub = X_values[valid_index], y_values[valid_index]

            lgbm_cl.fit(X_train_sub, y_train_sub, eval_set=[(X_valid_sub, y_valid_sub)])
            y_pred = lgbm_cl.predict(X_valid_sub)
            y_pred_proba = lgbm_cl.predict_proba(X_valid_sub)

            accuracy = accuracy_score(y_valid_sub, y_pred)
            logloss = log_loss(y_valid_sub, y_pred_proba)

            accuracy_scores.append(accuracy)
            logloss_scores.append(logloss)

        mean_accuracy = np.mean(accuracy_scores)

        log_model_and_params(lgbm_cl, trial, params, mean_accuracy)

    return (
        mean_accuracy  # or return mean_accuracy depending on what you want to optimize
    )


def Optuna_flow(
    model_name, model_version, X, y, test_size=0.2, n_trials=30, random_state=42
):
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    study = optuna.create_study(study_name="test", direction="maximize")
    study.optimize(lambda trial: objective(X_train, y_train, trial), n_trials=n_trials)
    best_trial = study.best_trial
    best_params = best_trial.params
    with open("./output/best_param.json", "w") as outfile:
        json.dump(best_params, outfile)

    experiment_id = create_or_get_experiment("lightgbm-optuna")
    print(f"Experiment ID: {experiment_id}, Best Trial ID: {best_trial.number}")
    runs_df = mlflow.search_runs(
        experiment_ids=experiment_id,
        order_by=["metrics.mean_accuracy DESC"],
    )
    best_run = runs_df.iloc[0]
    best_run_id = best_run["run_id"]

    _ = mlflow.register_model("runs:/" + best_run_id + "/lightgbm_model", model_name)

    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    print(model)

    return model
