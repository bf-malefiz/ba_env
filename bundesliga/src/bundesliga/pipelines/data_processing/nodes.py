from collections import OrderedDict

import numpy as np
import pandas as pd

GOALS_HOME = "FTHG"
GOALS_AWAY = "FTAG"


def build_team_lexicon(df: pd.DataFrame) -> pd.DataFrame:
    team1 = df["HomeTeam"].unique().astype("U")
    team2 = df["AwayTeam"].unique().astype("U")
    teams = np.unique(np.concatenate((team1, team2)))

    team_indices = OrderedDict()

    for i, t in enumerate(teams):
        team_indices[t] = i
    lex = pd.DataFrame(list(team_indices.items()), columns=["team", "index"])
    lex.set_index("team", inplace=True)
    return lex


def get_goal_results(df: pd.DataFrame, team_indices: pd.DataFrame):
    home_goals = list()
    away_goals = list()
    for _, r in df.iterrows():
        home_team = r.HomeTeam

        away_team = r.AwayTeam
        home_goals.append(
            (
                team_indices.loc[home_team, "index"],
                team_indices.loc[away_team, "index"],
                r[GOALS_HOME],
            )
        )
        away_goals.append(
            (
                team_indices.loc[home_team, "index"],
                team_indices.loc[away_team, "index"],
                r[GOALS_AWAY],
            )
        )

    goals = pd.DataFrame({"home_goals": home_goals, "away_goals": away_goals})

    return goals


def vectorize_data(goals) -> pd.DataFrame:
    home_id = np.array([hg[0] for hg in goals["home_goals"]])
    away_id = np.array([hg[1] for hg in goals["home_goals"]])
    home_goals = np.array([hg[2] for hg in goals["home_goals"]])
    away_goals = np.array([ag[2] for ag in goals["away_goals"]])
    toto = np.where(
        home_goals == away_goals, 0, np.where(home_goals > away_goals, 1, 2)
    )
    vectorized_data = pd.DataFrame(
        {
            "home_id": home_id,
            "away_id": away_id,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "toto": toto,
        }
    )

    return vectorized_data


def extract_features(
    vectorized_data: pd.DataFrame, parameters: pd.DataFrame
) -> pd.DataFrame:
    feature_data = vectorized_data[parameters["model_config"]["features"]]

    return feature_data


def extract_x_data(feature_data: pd.DataFrame, timeframe=None) -> pd.DataFrame:
    if not timeframe:
        x_data = feature_data
    else:
        raise NotImplementedError
    return x_data


def extract_y_data(
    vectorized_data: pd.DataFrame, parameters: pd.DataFrame, timeframe=None
) -> pd.Series:
    if not timeframe:
        y_data = vectorized_data[parameters["model_config"]["targets"]].apply(
            lambda row: (row["home_goals"], row["away_goals"]), axis=1
        )
        return y_data

    else:
        raise NotImplementedError
