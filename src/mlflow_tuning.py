import json
import pickle
import time
from typing import Any, Dict

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMModel
from loguru import logger
from mlflow.exceptions import MlflowException
from numpy import ndarray
from optuna.trial import Trial
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RepeatedKFold, train_test_split


def create_or_get_experiment(name: str) -> str:
    """
    Create or get an mlflow experiment based on the experiment name
    specified.

    Args:
        name (str): name to be given to the experiment
        or name of the experiment to be retrieved

    Raises:
        ValueError: if the experiment name is not found

    Returns:
        str: experiment ID in string format
    """
    try:
        experiment_id = mlflow.create_experiment(name)
    except MlflowException:
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is not None:
            experiment_id = experiment.experiment_id
        else:
            raise ValueError("Experiment not found.")
    return experiment_id


def log_model_and_params(
    model: LGBMModel, trial: Trial, params: Dict[str, Any], mean_accuracy: float
):
    """
    Log the model, params, and mean accuracy from mlflow experiments.

    Args:
        model (LGBMModel): the lightgbm trained every trial
        trial (Trial): the optuna trial
        params (Dict[str, Any]): the parameters used for the lightgbm model
        mean_accuracy (float): the mean accuracy of the model for every trial
    """
    mlflow.lightgbm.log_model(model, "lightgbm_model")
    mlflow.log_params(params)
    mlflow.log_metric("mean_accuracy", mean_accuracy)
    trial.set_user_attr(key="best_booster", value=pickle.dumps(model))


def objective(X_train: pd.DataFrame, y_train: ndarray, trial: Trial) -> float:
    """
    The objective function for the optuna study.

    Args:
        X_train (pd.DataFrame): predictors from training data
        y_train (ndarray): response from training data
        trial (Trial): the optuna trial

    Returns:
        float: accuracy score of the model
    """
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

            try:
                lgbm_cl.fit(
                    X_train_sub, y_train_sub, eval_set=[(X_valid_sub, y_valid_sub)]
                )
            except Exception as e:
                print(f"Failed to train model: {e}")
                return  # or break, depending on what you want to do when training fails

            try:
                y_pred = lgbm_cl.predict(X_valid_sub)
            except Exception as e:
                print(f"Failed to make predictions: {e}")
                return
            y_pred_proba = lgbm_cl.predict_proba(X_valid_sub)

            accuracy = accuracy_score(y_valid_sub, y_pred)
            logloss = log_loss(y_valid_sub, y_pred_proba)

            accuracy_scores.append(accuracy)
            logloss_scores.append(logloss)

        mean_accuracy = np.mean(accuracy_scores)

        log_model_and_params(lgbm_cl, trial, params, mean_accuracy)

    return mean_accuracy


def Optuna_flow(
    model_name: str,
    model_version: str,
    X: pd.DataFrame,
    y: ndarray,
    test_size: float = 0.2,
    n_trials: int = 30,
    random_state: int = 42,
    max_retries: int = 3,
    delay: int = 5,
) -> LGBMModel:
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if not isinstance(X, pd.DataFrame):
        raise ValueError("X should be a DataFrame")
    if not isinstance(y, np.ndarray):
        raise ValueError("y should be a ndarray")
    if not (0 < test_size < 1):
        raise ValueError("test_size should be a float between 0 and 1")
    if not (isinstance(n_trials, int) and n_trials > 0):
        raise ValueError("n_trials should be a positive integer")

    # assert isinstance(model_name, str), "model_name should be a string"
    # assert isinstance(model_version, str), "model_version should be a string"
    # assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
    # assert isinstance(y, np.ndarray), "y should be a ndarray"
    # assert 0 < test_size < 1, "test_size should be a float between 0 and 1"
    # assert (
    #     isinstance(n_trials, int) and n_trials > 0
    # ), "n_trials should be a positive integer"
    # assert isinstance(random_state, int), "random_state should be an integer"

    study = optuna.create_study(study_name="test", direction="maximize")

    for _ in range(max_retries):
        try:
            study.optimize(
                lambda trial: objective(X_train, y_train, trial), n_trials=n_trials
            )
            best_trial = study.best_trial
            best_params = best_trial.params
            break
        except Exception as e:
            print(f"An error occurred: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    else:
        raise RuntimeError("Failed to optimize the study after maximum retries")
    with open("./output/best_param.json", "w") as outfile:
        json.dump(best_params, outfile)

    experiment_id = create_or_get_experiment("lightgbm-optuna")
    runs_df = mlflow.search_runs(
        experiment_ids=experiment_id,
        order_by=["metrics.mean_accuracy DESC"],
    )
    best_run = runs_df.iloc[0]
    best_run_id = best_run["run_id"]

    try:
        _ = mlflow.register_model("runs:/" + best_run_id + "/lightgbm_model", model_name)
    except MlflowException as e:
        logger.error(f"Failed to register model: {e}")

    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    logger.info("Model loaded. the model information is as follows: {}".format(model))

    return model
