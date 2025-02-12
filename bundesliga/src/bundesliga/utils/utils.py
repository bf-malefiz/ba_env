"""
Module: utils.py

Summary:
    This module provides utility functions for splitting data by time (match matches),
    handling configuration loading and dictionary merging for the custom resolver and extracting goal-related
    distributions.

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


def calculate_day_accuracies(df: pd.DataFrame) -> list[float]:
    """Calculates the accuracy for each day in a DataFrame of match predictions.

    This helper function processes a DataFrame of match predictions in pairs of 9 rows and calculates the accuracy

    Args:
        df (pd.DataFrame): A DataFrame containing columns "predicted_result" and "ground_truth".

    Returns:
        list[float]: The mean accuracy across all days in the DataFrame.
    """
    day_accuracies = []

    for i in range(0, len(df) - 1, 9):
        # Select the current pair of rows
        day = df.iloc[i : i + 9]
        # Count the number of correct predictions in the pair
        correct = (day["predicted_result"] == day["ground_truth"]).sum()
        # Calculate accuracy for the pair divided by the number of matches (includes if there are less than 9 matches)
        accuracy = correct / len(day)
        day_accuracies.append(accuracy)

    return day_accuracies


def calculate_day_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the metrics for each day in a DataFrame.

    This helper function processes a DataFrame of match metrics in pairs of 9 rows and calculates the mean metric value

    Args:
        df (pd.DataFrame): A DataFrame containing columns of metrics.

    Returns:
        pd.DataFrame: The mean metrics across all days in the DataFrame.
    """
    day_metrics = []

    for i in range(0, len(df) - 1, 9):
        # Select the current pair of rows
        day = df.iloc[i : i + 9]

        day_metrics.append(day.mean())

    return pd.DataFrame(day_metrics)
