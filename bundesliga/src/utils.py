from copy import deepcopy
from pathlib import Path

import numpy as np
import omegaconf
import pandas as pd
import scipy.stats
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

min_mu = 0.0001
low = 10e-8  # Constant


def split_time_data(vectorized_data: pd.DataFrame, current_day):
    """
    Gibt train_X, train_y, test_X, test_y zurück,
    basierend auf der Spalte 'matchday' in X / y.
    """
    current_day = int(current_day)
    train_data = vectorized_data[:current_day]
    test_data = vectorized_data[current_day : current_day + 1]

    return train_data, test_data


def load_config():
    proj_path = Path(__file__).parent.parent
    conf_path = str(proj_path / settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path)

    return conf_loader


def merge_dicts(dict1, dict2):
    """
    Recursively merge two dictionaries.

    Args:
        dict1 (dict): The first dictionary to merge.
        dict2 (dict): The second dictionary to merge.

    Returns:
        dict: The merged dictionary.
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
    return model.team_lexicon


def get_goal_distribution(diff, max_goals=20):
    poisson_goals = np.zeros(max_goals)
    k = np.arange(0, max_goals)
    for lambda_ in diff:
        lambda_ = max(low, lambda_)
        poisson_goals += scipy.stats.poisson.pmf(k, lambda_)
    poisson_goals = poisson_goals / poisson_goals.sum()
    return poisson_goals
