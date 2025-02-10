"""
Module: utils.py

Summary:
    This module provides utility functions for splitting data by time (match matches),
    handling configuration loading and dictionary merging for the custom resolver, extracting goal-related
    distributions, and computing various scoring metrics (Brier Score, RPS, log-likelihood, RMSE, MAE)
    for evaluating model predictions in a football analytics context.

Dependencies:
    - copy.deepcopy
    - pathlib.Path
    - numpy as np
    - omegaconf
    - pandas as pd
    - scipy.stats
    - kedro.config.OmegaConfigLoader
    - kedro.framework.project.settings
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import omegaconf
import pandas as pd
import scipy.stats
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

from bundesliga.model.base_footballmodel import FootballModel

min_mu = 0.0001
low = 1e-8


def split_time_data(vectorized_data: pd.DataFrame, current_match: str | int) -> tuple:
    """
    Splits a DataFrame into training and testing subsets based on a given match index.

    This function separates the input DataFrame into a training set consisting of all rows
    before the current match and a test set consisting of the row corresponding to the current match.
    It assumes that the DataFrame is sorted in ascending order by match.

    Parameters:
        vectorized_data (pd.DataFrame): DataFrame containing match data.
        current_match (int): The index of the current match (if provided as a str, it will be converted to int).

    Returns:
        tuple: A tuple containing:
            - train_data (pd.DataFrame): Rows before the current match.
            - test_data (pd.DataFrame): The row(s) corresponding exactly to the current match.
    """
    current_match = int(current_match) + 1
    train_data = vectorized_data[:current_match]
    test_data = vectorized_data[current_match : current_match + 1]
    return train_data, test_data


def load_config() -> OmegaConfigLoader:
    """
    Loads a Kedro-style configuration using OmegaConf.

    This function constructs the configuration loader by determining the project's configuration
    directory based on the Kedro settings and returns an OmegaConfigLoader instance that can load
    and merge configuration files from that directory.

    Returns:
        OmegaConfigLoader: An object capable of loading and merging configuration files from the project's conf directory.
    """
    proj_path = Path(__file__).parent.parent.parent.parent
    conf_path = str(proj_path / settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path)

    return conf_loader


def merge_dicts(dict1, dict2):
    """
    Recursively merges two dictionaries or OmegaConf DictConfigs.

    If a key exists in both dictionaries and both corresponding values are DictConfigs,
    they are merged recursively. Otherwise, the value from the second dictionary overwrites
    the value from the first. A deep copy is used to avoid modifying the original dictionaries.

    Parameters:
        dict1 (dict or omegaconf.DictConfig): The first dictionary or configuration.
        dict2 (dict or omegaconf.DictConfig): The second dictionary or configuration, whose values take precedence.

    Returns:
        dict: A new dictionary containing the merged keys and values.
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


def get_teams(model: FootballModel) -> pd.DataFrame:
    """
    Retrieves the team lexicon from a model object.

    This function extracts and returns the 'team_lexicon' attribute from the given model,
    which contains a mapping of team names to unique indices.

    Parameters:
        model (FootballModel): A model object that holds a 'team_lexicon' attribute.

    Returns:
        pd.DataFrame: The team lexicon, as a DataFrame.
    """
    return model.team_lexicon


def get_goal_distribution(diff, max_goals: int = 20):
    """
    Computes a Poisson-based probability distribution over 0 to max_goals-1 goals.

    For each lambda value in 'diff', the function calculates the Poisson probability mass function (PMF),
    sums these probabilities element-wise, and then normalizes the result to form a valid probability distribution. (unused)

    Parameters:
        diff (iterable of float): Lambda values for Poisson distributions, each clamped to a minimum threshold.
        max_goals (int, optional): The maximum number of goals to consider (default is 20).

    Returns:
        np.ndarray: A normalized array of length 'max_goals' representing the combined Poisson probabilities.
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


def get_probability(row: pd.Series) -> float:
    """
    Extracts the probability of the true outcome from a DataFrame row.

    Given a row that includes predicted probabilities for 'home', 'away', and 'tie', along with the actual outcome
    specified in 'true_result', this function returns the probability corresponding to the actual outcome.

    Parameters:
        row (pd.Series): A row containing keys 'home', 'away', 'tie', and 'true_result'.

    Returns:
        float: The predicted probability for the true outcome.
    """
    # Define the order of outcomes corresponding to 0, 1, 2
    outcomes = ["home", "away", "tie"]
    # Get the column name corresponding to the true result (0 => 'home', 1 => 'away', 2 => 'tie')
    true_column = outcomes[int(row["true_result"])]
    return row[true_column]


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
