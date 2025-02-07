"""
Module: evaluation.py

Summary:
    This module provides functions to evaluate football match prediction models.
    It includes functions to evaluate individual match predictions, aggregate metrics across datasets and models,
    and determine match outcomes.

Dependencies:
    - numpy as np
    - pandas as pd
    - bundesliga.utils.utils: brier_score, log_likelihood, rmse_mae, rps
    - bundesliga.utils.validation: validate_dataframe
"""

import pandas as pd

from bundesliga.model.base_footballmodel import FootballModel
from bundesliga.utils.utils import brier_score, log_likelihood, rmse_mae, rps
from bundesliga.utils.validation import validate_dataframe


@validate_dataframe(
    df_arg_name="test_data",
    required_columns=["home_goals", "away_goals"],
    allow_empty=False,
)
@validate_dataframe(
    df_arg_name="predictions",
    required_columns=["home_goals", "away_goals"],
    allow_empty=False,
)
def evaluate_match(
    model: FootballModel, test_data: pd.DataFrame, predictions: pd.DataFrame
) -> dict:
    """
    Evaluates the performance of a football match prediction model for a single match.

    This function compares the model's predictions against the true match data by computing several
    evaluation metrics including negative log likelihood, brier score, RPS, and error metrics (RMSE and MAE)
    for both home and away teams. It also determines the categorical match outcomes (e.g., "home", "away", "tie")
    for both the ground truth and predictions.

    Args:
        model (FootballModel): The prediction model that provides a method to predict outcome probabilities.
        test_data (pd.DataFrame): A DataFrame containing true goal results with columns "home_goals" and "away_goals".
        predictions (pd.DataFrame): A DataFrame containing the model's predicted goals with columns "home_goals" and "away_goals".

    Returns:
        dict: A dictionary containing evaluation metrics such as ground truth result, predicted result, predicted probabilities,
              RMSE and MAE for home and away, negative log likelihood, brier score, and RPS.
    """
    test_goal = test_data[["home_goals", "away_goals"]]
    test_goal = test_goal.copy()
    test_goal["true_result"] = test_goal.apply(true_result, axis=1)

    toto_probs = model.predict_toto_probabilities(predictions)
    toto_probs = pd.DataFrame(toto_probs, columns=["home", "away", "tie"])
    toto_probs["predicted_result"] = toto_probs.apply(predicted_result, axis=1)

    test_goal = test_goal.reset_index(drop=True)
    toto_probs = toto_probs.reset_index(drop=True)

    # Additional evaluation metric examples
    negative_log_likelihood = log_likelihood(toto_probs, test_goal)
    brier_score_ = brier_score(test_goal, toto_probs)
    rps_mean = rps(test_goal, toto_probs)
    rmse_home, mae_home, rmse_away, mae_away = rmse_mae(predictions, test_data)

    results = {
        "ground_truth": test_goal["true_result"].values[0],
        "predicted_result": toto_probs["predicted_result"].values[0],
        "home_prob": toto_probs["home"].values[0],
        "away_prob": toto_probs["away"].values[0],
        "tie_prob": toto_probs["tie"].values[0],
        "toto_probs": toto_probs,
        "rmse_home": rmse_home,
        "mae_home": mae_home,
        "rmse_away": rmse_away,
        "mae_away": mae_away,
        "neg_log_likelihood": negative_log_likelihood,
        "brier_score": brier_score_,
        "rps": rps_mean,
    }
    return results


def aggregate_dataset_metrics(
    *kwargs,
):
    """
    Aggregates match evaluation metrics across a dataset.

    This function aggregates evaluation metrics provided as match metric dictionaries by computing their mean values.
    It also constructs a nested run name for tracking purposes. Metrics such as accuracy, RMSE, MAE, negative log likelihood,
    brier score, and RPS are aggregated.

    Args:
        *kwargs: A variable number of arguments where:
            - The first argument is a dictionary with model definitions (e.g., engine, dataset_name, variant, seed).
            - The subsequent arguments are dictionaries representing match evaluation metrics.

    Returns:
        tuple: A tuple containing:
            - mean_metrics (dict): A dictionary of aggregated mean metrics.
            - nested_run_name (str): A string for the nested run name, incorporating model definition details.
    """

    model_definitions = kwargs[0]
    engine = model_definitions["engine"]
    dataset_name = model_definitions["dataset_name"]
    variant = model_definitions["variant"]
    seed = model_definitions["seed"]
    all_match_metrics = pd.DataFrame(kwargs[1:])

    nested_run_name = (
        f"Dataset Metrics | engine={engine} | model={variant} | season={dataset_name}"
    )

    all_match_metrics["correct"] = all_match_metrics.apply(
        lambda row: row["predicted_result"] == row["ground_truth"], axis=1
    )

    # Now compute the mean for each metric. You can either compute them individually:
    mean_accuracy = all_match_metrics["correct"].mean()
    mean_rmse_home = all_match_metrics["rmse_home"].mean()
    mean_mae_home = all_match_metrics["mae_home"].mean()
    mean_rmse_away = all_match_metrics["rmse_away"].mean()
    mean_mae_away = all_match_metrics["mae_away"].mean()
    mean_neg_log_likelihood = all_match_metrics["neg_log_likelihood"].mean()
    mean_brier_score = all_match_metrics["brier_score"].mean()
    mean_rps = all_match_metrics["rps"].mean()

    mean_metrics = {
        "accuracy": mean_accuracy,
        "rmse_home": mean_rmse_home,
        "mae_home": mean_mae_home,
        "rmse_away": mean_rmse_away,
        "mae_away": mean_mae_away,
        "neg_log_likelihood": mean_neg_log_likelihood,
        "brier_score": mean_brier_score,
        "rps": mean_rps,
    }

    return mean_metrics, nested_run_name


def aggregate_model_metrics(
    *kwargs,
) -> tuple[dict, str]:
    """
    Aggregates evaluation metrics across multiple datasets for a given model.

    This function collects model metrics from various datasets and computes the mean values for each metric.
    It also generates a nested run name to track the overall model evaluation metrics. Aggregated metrics include
    accuracy, RMSE, MAE, negative log likelihood, brier score, and RPS.

    Args:
        *kwargs: A variable number of arguments where:
            - The first argument is a dictionary containing model definitions (e.g., engine, variant, seed).
            - The subsequent arguments are dictionaries representing aggregated metrics from individual datasets.

    Returns:
        tuple: A tuple containing:
            - mean_model_metrics (dict): A dictionary of aggregated mean metrics across datasets.
            - nested_run_name (str): A string representing the nested run name for model metrics tracking.
    """

    model_definitions = kwargs[0]
    engine = model_definitions["engine"]
    variant = model_definitions["variant"]
    seed = model_definitions["seed"]
    all_dataset_metrics = pd.DataFrame(kwargs[1:])

    nested_run_name = f"Model Metrics | engine={engine} | model={variant} "
    mean_accuracy = all_dataset_metrics["accuracy"].mean()
    mean_rmse_home = all_dataset_metrics["rmse_home"].mean()
    mean_mae_home = all_dataset_metrics["mae_home"].mean()
    mean_rmse_away = all_dataset_metrics["rmse_away"].mean()
    mean_mae_away = all_dataset_metrics["mae_away"].mean()
    mean_neg_log_likelihood = all_dataset_metrics["neg_log_likelihood"].mean()
    mean_brier_score = all_dataset_metrics["brier_score"].mean()
    mean_rps = all_dataset_metrics["rps"].mean()

    mean_model_metrics = {
        "accuracy": mean_accuracy,
        "rmse_home": mean_rmse_home,
        "mae_home": mean_mae_home,
        "rmse_away": mean_rmse_away,
        "mae_away": mean_mae_away,
        "neg_log_likelihood": mean_neg_log_likelihood,
        "brier_score": mean_brier_score,
        "rps": mean_rps,
    }
    return mean_model_metrics, nested_run_name


@validate_dataframe(
    df_arg_name="predictions",
    required_columns=["home_goals", "away_goals"],
    allow_empty=False,
)
def determine_winner(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the winner of a match based on predicted goal values.

    This function evaluates each match by comparing predicted home and away goals, and assigns a match outcome
    ("home", "away", or "tie") accordingly. The result is appended as a new column "winner" to the predictions DataFrame.

    Args:
        predictions (pd.DataFrame): A DataFrame containing predicted goals with columns "home_goals" and "away_goals".

    Returns:
        pd.DataFrame: The modified DataFrame including a new "winner" column indicating the predicted match outcome.
    """

    def get_winner(row):
        if row["home_goals"] > row["away_goals"]:
            return "home"
        elif row["home_goals"] < row["away_goals"]:
            return "away"
        else:
            return "tie"

    predictions["winner"] = predictions.apply(get_winner, axis=1)
    return predictions


def true_result(df_row):
    """
    Determines the true match result from a row of actual goal data.

    This helper function compares the home and away goal values in a DataFrame row and returns the actual match outcome.

    Args:
        df_row (pd.Series): A row from a DataFrame containing "home_goals" and "away_goals".

    Returns:
        str: "home" if the home team wins, "away" if the away team wins, or "tie" if both teams score equally.
    """
    # todo: could be simplified by using the column from the dataframe directly
    if df_row["home_goals"] > df_row["away_goals"]:
        return "home"
    elif df_row["home_goals"] < df_row["away_goals"]:
        return "away"
    else:
        return "tie"


def predicted_result(probs_row):
    """
    Determines the predicted match result based on outcome probabilities.

    This helper function takes a row of predicted probabilities and returns the match outcome corresponding
    to the highest probability.

    Args:
        probs_row (pd.Series): A row containing predicted probabilities with keys "home", "away", and "tie".

    Returns:
        str: The predicted outcome ("home", "away", or "tie") based on the maximum probability.
    """
    # probs_row = { "home": ..., "away": ..., "tie": ... }
    return probs_row.idxmax()
