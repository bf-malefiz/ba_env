import pandas as pd
import numpy as np
from collections import OrderedDict

GOALS_HOME='FTHG'
GOALS_AWAY='FTAG'

def _get_teams(df: pd.DataFrame) -> np.array:
    t1 = df.HomeTeam.unique().astype('U')
    t2 = df.AwayTeam.unique().astype('U')
    teams = np.unique(np.concatenate((t1,t2)))

    assert len(teams) == 18
    #nb_teams = len(teams)

    return teams

def _build_team_lexicon(teams: np.array) -> OrderedDict:
    team_indices = OrderedDict()
    for i, t in enumerate(teams):
        team_indices[t] = i
    return team_indices


def _get_goal_results(df: pd.DataFrame, team_indices: np.array):
    home_goals = list()
    away_goals = list()
    for index, r in df.iterrows():
        home_team = r.HomeTeam
        away_team = r.AwayTeam
        goals=r[GOALS_HOME]
        home_goals.append((team_indices[home_team], team_indices[away_team], goals))
    for index, r in df.iterrows():
        home_team = r.HomeTeam
        away_team = r.AwayTeam
        goals=r[GOALS_AWAY]
        away_goals.append((team_indices[home_team], team_indices[away_team], goals))
    
    return home_goals, away_goals


def _vectorized_data(home_goals_, away_goals_):
    home_id = np.array([hg[0] for hg in home_goals_])
    away_id = np.array([hg[1] for hg in home_goals_])
    home_goals = np.array([hg[2] for hg in home_goals_])
    away_goals = np.array([ag[2] for ag in away_goals_])
    toto = np.where(home_goals == away_goals, 0,
                        np.where(home_goals > away_goals, 1, 2))
    return home_id, away_id, home_goals, away_goals, toto


def preprocess_league_data(df: pd.DataFrame) -> pd.DataFrame:
    teams = _get_teams(df)
    team_indices = _build_team_lexicon(teams=teams)
    home_goals_, away_goals_ = _get_goal_results(df=df, team_indices=team_indices)

    # list of tuples
    # (home_team-index, away_team_index, scored_goals of home team resp. away team)

    return pd.DataFrame(_vectorized_data(home_goals_, away_goals_))