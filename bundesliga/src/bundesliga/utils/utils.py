"""
This module provides utility functions for splitting data by time (match matchs),
handling configuration loading and dictionary merging for the custom resolver, extracting goal-related
distributions, and computing various scoring metrics (Brier Score, RPS, log-
likelihood, RMSE, MAE) for evaluating model predictions in a football analytics
context.

Notes:
------
- Many of these functions rely on well-formed Pandas DataFrames with specific column
  names (e.g., 'true_result' for the actual match outcome, or 'home_goals'/'away_goals'
  for predicted vs. actual goal counts).
- The Poisson-based distribution functions assume strictly positive lambda values
  (`diff`), adjusting by a small constant if lambda is extremely small.

Author:
    Philipp Dahlke

License:
    Specify your project license here.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import omegaconf
import pandas as pd
import scipy.stats
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

min_mu = 0.0001
low = 1e-8  # Adjusted from '10e-8' for clarity


def split_time_data(vectorized_data: pd.DataFrame, current_match: int):
    """
    Splits a DataFrame into training and testing subsets based on a given matchmatch.

    Parameters
    ----------
    vectorized_data : pd.DataFrame
        A DataFrame containing match data, typically indexed by match or matchmatch.
    current_match : int or str
        The current match index (converted to int if given as str). All rows
        before `current_match` form the training set, and the row(s) at
        `current_match` form the test set.

    Returns
    -------
    train_data : pd.DataFrame
        Subset of `vectorized_data` up to (but not including) the current match.
    test_data : pd.DataFrame
        The portion of `vectorized_data` corresponding exactly to `current_match`.

    Notes
    -----
    - This function assumes that `vectorized_data` is sorted by match (ascending)
      or that rows prior to `current_match` logically represent the training window.
    """
    current_match = int(current_match) + 1
    train_data = vectorized_data[:current_match]
    test_data = vectorized_data[current_match : current_match + 1]
    return train_data, test_data


def load_config():
    """
    Loads a Kedro-style configuration using OmegaConf.

    Returns
    -------
    conf_loader : OmegaConfigLoader
        An object capable of loading and merging configuration files
        from the project's conf directory.

    Notes
    -----
    - Relies on `settings.CONF_SOURCE` for the conf path inside the Kedro project.
    - Adjusts `proj_path` to be the parent of the current file, then appends `CONF_SOURCE`.
    """
    proj_path = Path(__file__).parent.parent.parent.parent
    conf_path = str(proj_path / settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path)

    return conf_loader


def merge_dicts(dict1, dict2):
    """
    Recursively merge two dictionaries or OmegaConf DictConfigs.

    If a key exists in both `dict1` and `dict2`, and both values are
    `omegaconf.DictConfig`, this function merges them recursively.
    Otherwise, the value in `dict2` overwrites the value in `dict1`.

    Parameters
    ----------
    dict1 : dict or omegaconf.DictConfig
        The first dictionary / config to merge.
    dict2 : dict or omegaconf.DictConfig
        The second dictionary / config, whose values take precedence.

    Returns
    -------
    dict
        A new dictionary-like object containing the merged keys/values.

    Notes
    -----
    - Uses `deepcopy` to avoid modifying the input dictionaries in-place.
    """
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if (
            key in result
            and isinstance(result[key], omegaconf.dictconfig.DictConfig)
            and isinstance(value, omegaconf.dictconfig.DictConfig)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def get_teams(model):
    """
    Retrieves the `team_lexicon` attribute from a model object.

    Parameters
    ----------
    model : Any
        An object (usually a trained model) that holds a `team_lexicon`
        attribute.

    Returns
    -------
    team_lexicon : Any
        Whatever is stored in `model.team_lexicon` (often a DataFrame or dict).
    """
    return model.team_lexicon


def get_goal_distribution(diff, max_goals=20):
    """
    Computes a Poisson-based probability distribution over 0..max_goals-1 goals,
    given a set of Poisson rate parameters (lambdas) in `diff`.

    The function first calculates a Poisson PMF for each lambda and sums them,
    then normalizes the resulting distribution.

    Parameters
    ----------
    diff : iterable of float
        A collection of lambda values for Poisson distributions. Each value
        is clamped to a minimum of `low` to avoid zero or negative rates.
    max_goals : int, optional
        The maximum number of goals to consider (default is 20).

    Returns
    -------
    poisson_goals : np.ndarray
        A normalized array of length `max_goals` representing the combined
        Poisson probabilities across all lambdas.

    Notes
    -----
    - If multiple lambdas are provided, their PMFs are summed element-wise
      and then re-normalized to sum to 1.
    """
    poisson_goals = np.zeros(max_goals)
    k = np.arange(0, max_goals)
    for lambda_ in diff:
        lambda_ = max(low, lambda_)
        poisson_goals += scipy.stats.poisson.pmf(k, lambda_)
    # Normalize
    poisson_goals = poisson_goals / poisson_goals.sum()
    return poisson_goals


def brier_score(test_goal: pd.DataFrame, toto_probs: pd.DataFrame) -> float:
    """
    Computes the Brier Score (multi-class version) for match outcomes: home, away, tie.

    Parameters
    ----------
    test_goal : pd.DataFrame
        Must contain a column 'true_result' indicating the actual match outcome
        ('home', 'away', or 'tie').
    toto_probs : pd.DataFrame
        Predicted probabilities with columns ['home', 'away', 'tie'].

    Returns
    -------
    float
        The mean Brier Score over the rows of data.

    Notes
    -----
    - The Brier Score is the mean squared error between the one-hot encoded
      true outcome and the predicted probability distribution.
    - If test_goal has fewer than 3 categories, reindexing ensures
      columns 'home', 'away', and 'tie' exist, filling missing categories
      with 0. This might happen if all matches in `test_goal` had the same result.
    """
    # Convert true_result to one-hot
    one_hot = pd.get_dummies(test_goal["true_result"])
    one_hot = one_hot.reindex(columns=["home", "away", "tie"], fill_value=0)

    brier_per_game = ((one_hot - toto_probs[["home", "away", "tie"]]) ** 2).sum(axis=1)
    return brier_per_game.mean()


def rps(test_goals: pd.DataFrame, toto_probs: pd.DataFrame) -> float:
    """
    Calculates the mean Ranked Probability Score (RPS) for a set of matches
    with possible outcomes 'home', 'away', or 'tie'.

    Parameters
    ----------
    test_goals : pd.DataFrame
        Contains a column 'true_result' with the actual outcome per match.
    toto_probs : pd.DataFrame
        Contains columns ['home', 'away', 'tie'] with predicted probabilities.

    Returns
    -------
    float
        The average RPS across all matches.

    Notes
    -----
    - The function applies `rps_3` row by row, passing the predicted
      probabilities for 'away', 'tie', 'home', and the actual result.
    - RPS is a measure of how well-structured the probability distribution
      is, taking into account the ordinal nature of outcomes if applicable.
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
    Computes the Ranked Probability Score (RPS) for a single match with three
    possible outcomes (away, tie, home).

    Parameters
    ----------
    prob_away : float
        Predicted probability for 'away' outcome.
    prob_tie : float
        Predicted probability for 'tie' outcome.
    prob_home : float
        Predicted probability for 'home' outcome.
    actual : {'away', 'tie', 'home'}
        The actual observed outcome of the match.

    Returns
    -------
    float
        The RPS value for this single match.

    Notes
    -----
    - RPS is computed by first forming cumulative probabilities in the
      order (away, tie, home). Then, each partial sum is compared to
      the true outcome's partial sum. The squared differences are summed.
    - The order assumed here is away < tie < home. If `actual == 'away'`,
      that implies O0=1, O1=1, else if `actual == 'tie'`, then O0=0, O1=1, etc.
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
    Calculates the negative log-likelihood (NLL) for predicted match outcome
    probabilities vs. the true results.

    Parameters
    ----------
    toto_probs : pd.DataFrame
        Contains columns ['home', 'away', 'tie'] representing the predicted
        probabilities for each match.
    test_goals : pd.DataFrame
        Must have a column 'true_result' with values in {'home','away','tie'}.

    Returns
    -------
    float
        The average negative log-likelihood across all matches.

    Notes
    -----
    - A small epsilon (1e-12) is added to avoid log(0) issues.
    - The final result is -mean(log(prob_of_true_outcome)).
    """
    merged = pd.concat([toto_probs, test_goals["true_result"]], axis=1)
    EPS = 1e-12

    merged["prob_of_true_result"] = merged.apply(get_probability, axis=1)
    merged["log_prob"] = np.log(merged["prob_of_true_result"] + EPS)
    return -merged["log_prob"].mean()


def get_probability(row: pd.Series) -> float:
    """
    Extracts the probability of the true outcome from a row containing
    'home', 'away', 'tie', and 'true_result' columns.

    Parameters
    ----------
    row : pd.Series
        Must include:
          - 'home': float
          - 'away': float
          - 'tie':  float
          - 'true_result': str in {'home','away','tie'}

    Returns
    -------
    float
        Probability of the actual outcome (row['true_result']).
    """
    return row[row["true_result"]]


def rmse_mae(predictions: pd.DataFrame, test_data: pd.DataFrame):
    """
    Calculates RMSE and MAE for predicted vs. actual goals in home and away scenarios.

    Parameters
    ----------
    predictions : pd.DataFrame
        A DataFrame with columns ['home_goals', 'away_goals'] representing
        predicted goals.
    test_data : pd.DataFrame
        A DataFrame with columns ['home_goals', 'away_goals'] representing
        actual goals. Typically contains a single row or consistent values
        for the test match(es).

    Returns
    -------
    (float, float, float, float)
        (rmse_home, mae_home, rmse_away, mae_away), where:
          - rmse_home : float
              RMSE for home goals
          - mae_home : float
              MAE for home goals
          - rmse_away : float
              RMSE for away goals
          - mae_away : float
              MAE for away goals

    Notes
    -----
    - Both `predictions` and `test_data` are reset_index to avoid alignment
      issues when subtracting Series.
    - This function assumes that for each index i in `predictions`, the
      corresponding row in `test_data` has the relevant actual goals.
      If `test_data` has only one row, it is reused.
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
