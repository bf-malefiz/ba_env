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

# def extract_features(
#     vectorized_data: pd.DataFrame,
# ) -> pd.DataFrame:
#     feature_data = vectorized_data[["home_id", "away_id"]]

#     return feature_data


# def extract_x_data(feature_data: pd.DataFrame, timeframe=None) -> pd.DataFrame:
#     if not timeframe:
#         x_data = feature_data
#     else:
#         raise NotImplementedError
#     return x_data


# def extract_y_data(vectorized_data: pd.DataFrame, timeframe=None) -> pd.DataFrame:
#     if not timeframe:
#         y_data = vectorized_data[["home_goals", "away_goals"]]
#         return y_data

#     else:
#         raise NotImplementedError


# def extract_toto(vectorized_data: pd.DataFrame) -> pd.DataFrame:
#     return pd.DataFrame(vectorized_data["toto"], columns=["toto"])


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


def get_diffs(
    team_1, team_2, offence, defence
):  # toDO: offenc defence add in function from model1
    diff_ij = offence[team_1] - defence[team_2]
    diff_ji = offence[team_2] - defence[team_1]
    return diff_ij, diff_ji


def get_probs_winner(teamIndex1, teamIndex2):
    diff_ij, diff_ji = get_diffs(teamIndex1, teamIndex2)
    diff_ij[diff_ij < min_mu] = min_mu
    diff_ji[diff_ji < min_mu] = min_mu

    goals_of_team_1 = np.array([np.random.poisson(r) for r in diff_ij])
    goals_of_team_2 = np.array([np.random.poisson(r) for r in diff_ji])

    team1_wins = goals_of_team_1 > goals_of_team_2
    team2_wins = goals_of_team_1 < goals_of_team_2
    tie = goals_of_team_1 == goals_of_team_2

    p1 = team1_wins.mean()
    p2 = team2_wins.mean()
    tie = tie.mean()
    np.testing.assert_almost_equal(1.0, p1 + tie + p2)
    return tie, p1, p2


def get_goal_distribution(diff, max_goals=20):
    poisson_goals = np.zeros(max_goals)
    k = np.arange(0, max_goals)
    for lambda_ in diff:
        lambda_ = max(low, lambda_)
        poisson_goals += scipy.stats.poisson.pmf(k, lambda_)
    poisson_goals = poisson_goals / poisson_goals.sum()
    return poisson_goals
