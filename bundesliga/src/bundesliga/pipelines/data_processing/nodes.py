from collections import OrderedDict

import numpy as np
import pandas as pd

GOALS_HOME = "FTHG"
GOALS_AWAY = "FTAG"


def _get_teams(df: pd.DataFrame) -> np.array:
    team1 = df["HomeTeam"].unique().astype("U")
    team2 = df["AwayTeam"].unique().astype("U")
    teams = np.unique(np.concatenate((team1, team2)))

    assert len(teams) == 18
    # nb_teams = len(teams)

    return teams


def _build_team_lexicon(teams: np.array) -> pd.DataFrame:
    team_indices = OrderedDict()
    for i, t in enumerate(teams):
        team_indices[t] = i
    lex = pd.DataFrame(list(team_indices.items()), columns=["team", "index"])
    lex.set_index("team", inplace=True)
    return lex


def _get_goal_results(df: pd.DataFrame, team_indices: pd.DataFrame):
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

    return home_goals, away_goals


def _vectorized_data(home_goals_, away_goals_) -> pd.DataFrame:
    home_id = np.array([hg[0] for hg in home_goals_])
    away_id = np.array([hg[1] for hg in home_goals_])
    home_goals = np.array([hg[2] for hg in home_goals_])
    away_goals = np.array([ag[2] for ag in away_goals_])
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


def preprocess_league_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    teams = _get_teams(df)
    team_indices = _build_team_lexicon(teams=teams)
    home_goals_, away_goals_ = _get_goal_results(df=df, team_indices=team_indices)

    # list of tuples
    # (home_team-index, away_team_index, scored_goals of home team resp. away team)

    return _vectorized_data(home_goals_, away_goals_), team_indices


def create_model_input_data(df: pd.DataFrame) -> pd.DataFrame:
    # merge if we use more then D1 Bundesliga data
    model_input_table = df
    return model_input_table
