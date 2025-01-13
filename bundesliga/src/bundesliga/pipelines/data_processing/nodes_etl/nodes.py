from collections import OrderedDict

import numpy as np
import pandas as pd

GOALS_HOME = "FTHG"
GOALS_AWAY = "FTAG"


def build_team_lexicon(df: pd.DataFrame) -> pd.DataFrame:
    home_team = df["HomeTeam"].unique().astype("U")
    # team2 = df["AwayTeam"].unique().astype("U") toDO: check if this is necessary

    team_indices = enumerate(np.unique(home_team))
    lex = pd.DataFrame(team_indices, columns=["index", "team"]).set_index("team")
    return lex


def get_goal_results(df: pd.DataFrame, lexicon: pd.DataFrame):
    home_goals = list()
    away_goals = list()
    for _, r in df.iterrows():
        home_team_name = r.HomeTeam
        away_team_name = r.AwayTeam

        home_goals.append(
            (
                lexicon.index.get_loc(home_team_name),
                lexicon.index.get_loc(away_team_name),
                r[GOALS_HOME],
            )
        )
        away_goals.append(
            (
                lexicon.index.get_loc(home_team_name),
                lexicon.index.get_loc(away_team_name),
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
    vectorized_data: pd.DataFrame,
) -> pd.DataFrame:
    feature_data = vectorized_data[["home_id", "away_id"]]

    return feature_data


def extract_x_data(feature_data: pd.DataFrame, timeframe=None) -> pd.DataFrame:
    if not timeframe:
        x_data = feature_data
    else:
        raise NotImplementedError
    return x_data


def extract_y_data(vectorized_data: pd.DataFrame, timeframe=None) -> pd.DataFrame:
    if not timeframe:
        y_data = vectorized_data[["home_goals", "away_goals"]]
        return y_data

    else:
        raise NotImplementedError


def extract_toto(vectorized_data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(vectorized_data["toto"], columns=["toto"])
