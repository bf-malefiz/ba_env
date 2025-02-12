"""
Module: utils.py

Summary:
    This module provides various scoring metrics (i.e. Brier Score, RPS, log-likelihood, RMSE, MAE)
    for evaluating model predictions in a football analytics context.

Dependencies:
    - numpy as np
    - pandas as pd
    - bundesliga.utils.utils.get_probability
"""

import numpy as np
import pandas as pd

from bundesliga.utils.utils import get_probability


def rmse_mae(
    predictions: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[float, float, float, float]:
    """
    Calculates RMSE and MAE for predicted versus actual goals in home and away scenarios.

    The function computes the root mean squared error (RMSE) and mean absolute error (MAE) separately for home
    and away goals by comparing the predictions against the actual values provided in test_data. Both inputs are
    reindexed to ensure proper alignment.

    Parameters:
        predictions (pd.DataFrame): DataFrame with predicted goals in columns ['home_goals', 'away_goals'].
        test_data (pd.DataFrame): DataFrame with actual goals in columns ['home_goals', 'away_goals'].

    Returns:
        tuple[float, float, float, float]: A tuple containing four floats:
            - rmse_home: RMSE for home goals.
            - mae_home: MAE for home goals.
            - rmse_away: RMSE for away goals.
            - mae_away: MAE for away goals.
    """
    predictions = predictions.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    errors_home = predictions["home_goals"].values - test_data["home_goals"].values[0]
    rmse_home = np.sqrt(np.mean(errors_home**2))
    mae_home = np.mean(np.abs(errors_home))

    errors_away = predictions["away_goals"].values - test_data["away_goals"].values[0]
    rmse_away = np.sqrt(np.mean(errors_away**2))
    mae_away = np.mean(np.abs(errors_away))

    return rmse_home, mae_home, rmse_away, mae_away


def brier_score(test_goal: pd.DataFrame, toto_probs: pd.DataFrame) -> float:
    """
    Computes the mean Brier Score for multi-class match outcomes.

    The Brier Score is calculated as the mean squared error between the one-hot encoded true outcomes
    and the predicted probability distributions for outcomes ('home', 'away', 'tie'). If certain categories
    are missing in the true outcomes, they are filled with zeros.

    Parameters:
        test_goal (pd.DataFrame): DataFrame containing the true match outcomes in a column 'true_result'.
        toto_probs (pd.DataFrame): DataFrame with predicted probabilities for outcomes in columns ['home', 'away', 'tie'].

    Returns:
        float: The average Brier Score computed over all matches.
    """
    # Convert true_result to one-hot
    one_hot = pd.get_dummies(test_goal["true_result"])
    one_hot = one_hot.reindex(columns=["home", "away", "tie"], fill_value=0)

    brier_per_game = ((one_hot - toto_probs[["home", "away", "tie"]]) ** 2).sum(axis=1)
    return brier_per_game.mean()


def rps(test_goals: pd.DataFrame, toto_probs: pd.DataFrame) -> float:
    """
    Calculates the average Ranked Probability Score (RPS) for a set of matches.

    The RPS measures the quality of probabilistic predictions by comparing the cumulative predicted probabilities
    against the cumulative true outcomes for each match. The function applies the single-match RPS calculation
    (using rps_3) across all matches and returns the mean score.

    Parameters:
        test_goals (pd.DataFrame): DataFrame containing the actual outcomes in a column 'true_result'.
        toto_probs (pd.DataFrame): DataFrame with predicted probabilities in columns ['home', 'away', 'tie'].

    Returns:
        float: The mean RPS across all matches.
    """
    rps_values = []
    for i in range(len(test_goals)):
        actual = test_goals["true_result"].iloc[i]
        p_away = toto_probs["away"].iloc[i]
        p_tie = toto_probs["tie"].iloc[i]
        p_home = toto_probs["home"].iloc[i]
        rps_val = rps_3(p_away, p_tie, p_home, actual)
        rps_values.append(rps_val)
    return np.mean(rps_values)


def rps_3(prob_away: float, prob_tie: float, prob_home: float, actual: str) -> float:
    """
    Computes the Ranked Probability Score (RPS) for a single match with three outcomes.

    The RPS is calculated by comparing the cumulative predicted probabilities with the cumulative true outcomes.
    The function assumes an ordinal order of outcomes (away < tie < home) and sums the squared differences
    of the cumulative distributions.

    Parameters:
        prob_away (float): Predicted probability for an away win.
        prob_tie (float): Predicted probability for a tie.
        prob_home (float): Predicted probability for a home win.
        actual (str): The actual outcome ('away', 'tie', or 'home').

    Returns:
        float: The RPS value for the single match.
    """
    # Cumulative predicted distribution
    F0 = prob_away
    F1 = prob_away + prob_tie

    # Determine cumulative true outcome
    if actual == "away":
        O0, O1 = 1.0, 1.0
    elif actual == "tie":
        O0, O1 = 0.0, 1.0
    else:  # 'home'
        O0, O1 = 0.0, 0.0

    return (F0 - O0) ** 2 + (F1 - O1) ** 2


def log_likelihood(toto_probs: pd.DataFrame, test_goals: pd.DataFrame) -> float:
    """
    Calculates the negative log-likelihood for predicted match outcome probabilities.

    This function computes the negative log-likelihood by extracting the probability assigned to the true outcome
    for each match, applying a small epsilon to prevent log(0), and returning the negative mean of the log probabilities.

    Parameters:
        toto_probs (pd.DataFrame): DataFrame with predicted probabilities in columns ['home', 'away', 'tie'].
        test_goals (pd.DataFrame): DataFrame containing the true match outcomes in a column 'true_result'.

    Returns:
        float: The average negative log-likelihood across all matches.
    """
    merged = pd.concat([toto_probs, test_goals["true_result"]], axis=1)
    EPS = 1e-12

    merged["prob_of_true_result"] = merged.apply(get_probability, axis=1)
    merged["log_prob"] = np.log(merged["prob_of_true_result"] + EPS)
    return -merged["log_prob"].mean()
