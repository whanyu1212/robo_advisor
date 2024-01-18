import pickle

import mlflow

# from sklearn.model_selection import train_test_split
import mlflow.pyfunc
import mlflow.xgboost
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold

# from mlflow import MlflowClient


def objective(X_train, y_train, trial, _best_auc=0):
    try:
        EXPERIMENT_ID = mlflow.create_experiment("lightgbm-optuna")
    except Exception:
        try:
            experiment = mlflow.get_experiment_by_name("lightgbm-optuna")
            if experiment is not None:
                EXPERIMENT_ID = experiment["experiment_id"]
            else:
                print("Experiment not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    with mlflow.start_run(experiment_id=EXPERIMENT_ID, nested=True):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
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
        aucpr_scores = []

        X_values = X_train.values
        y_values = y_train.values
        for train_index, valid_index in rkf.split(X_train):
            X_train_sub, y_train_sub = X_values[train_index], y_values[train_index]
            X_valid_sub, y_valid_sub = X_values[valid_index], y_values[valid_index]

            lgbm_cl.fit(
                X_train_sub,
                y_train_sub,
                eval_set=[(X_valid_sub, y_valid_sub)],
                verbose=0,
            )
            y_pred = lgbm_cl.predict(X_valid_sub)
            aucpr = roc_auc_score(y_valid_sub, y_pred)
            aucpr_scores.append(aucpr)

        mean_aucpr = np.mean(aucpr_scores)

        mlflow.lightgbm.log_model(lgbm_cl, "lightgbm_model")
        mlflow.log_param("Optuna_trial_num", trial.number)
        mlflow.log_params(params)

        if mean_aucpr > _best_auc:
            _best_auc = mean_aucpr
            with open("./output/trial_%d.pkl" % trial.number, "wb") as f:
                pickle.dump(lgbm_cl, f)
            mlflow.log_artifact("./output/trial_%d.pkl" % trial.number)
        mlflow.log_metric("VAL_AUC", mean_aucpr)
    return mean_aucpr
