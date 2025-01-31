from collections import OrderedDict

import numpy as np
import pandas as pd

GOALS_HOME = "FTHG"
GOALS_AWAY = "FTAG"


def build_team_lexicon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a team lexicon (mapping of team names to unique indices) from the input DataFrame.

    This function extracts unique team names from the "HomeTeam" column and assigns each team
    a unique index. The resulting lexicon is a DataFrame with team names as the index and
    their corresponding indices as values.

    Args:
        df (pd.DataFrame): The input DataFrame containing match data with a "HomeTeam" column.

    Returns:
        pd.DataFrame: A DataFrame representing the team lexicon, with team names as the index
                      and their corresponding indices as values.
    """
    if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        raise ValueError(
            "Input DataFrame must contain 'HomeTeam' and 'AwayTeam' columns."
        )

    # Combine home and away teams and remove duplicates
    all_teams = (
        pd.concat([df["HomeTeam"], df["AwayTeam"]])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    team_indices = enumerate(all_teams)
    lex = pd.DataFrame(team_indices, columns=["index", "team"]).set_index("team")
    return lex


def get_goal_results(df: pd.DataFrame, lexicon: pd.DataFrame):
    """
    Extracts goal results from the input DataFrame and maps team names to their indices.

    This function iterates over the input DataFrame and maps the home and away team names
    to their corresponding indices using the provided team lexicon. It returns a DataFrame
    containing tuples of (home_team_index, away_team_index, goals) for each match.

    Args:
        df (pd.DataFrame): The input DataFrame containing match data with "HomeTeam" and "AwayTeam" columns.
        lexicon (pd.DataFrame): The team lexicon mapping team names to indices.

    Returns:
        pd.DataFrame: A DataFrame containing goal results with columns "home_goals" and "away_goals".
                      Each entry is a tuple of (home_team_index, away_team_index, goals).
    """
    required_columns = ["HomeTeam", "AwayTeam", GOALS_HOME, GOALS_AWAY]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Map team names to indices using the lexicon
    df["home_id"] = df["HomeTeam"].apply(lambda x: lexicon.index.get_loc(x))
    df["away_id"] = df["AwayTeam"].apply(lambda x: lexicon.index.get_loc(x))

    # Create goal results DataFrame
    goals = pd.DataFrame(
        {
            "home_goals": list(zip(df["home_id"], df["away_id"], df[GOALS_HOME])),
            "away_goals": list(zip(df["home_id"], df["away_id"], df[GOALS_AWAY])),
        }
    )
    return goals


def vectorize_data(goals) -> pd.DataFrame:
    """
    Vectorizes the goal results into a structured DataFrame.

    Args:
        goals (pd.DataFrame): A DataFrame containing goal results with columns "home_goals" and "away_goals".
                              Each entry is a tuple of (home_team_index, away_team_index, goals).

    Returns:
        pd.DataFrame: A structured DataFrame with columns:
                      - "home_id": Home team index.
                      - "away_id": Away team index.
                      - "home_goals": Goals scored by the home team.
                      - "away_goals": Goals scored by the away team.
                      - "toto": Match outcome (0 for draw, 1 for home win, 2 for away win).

    Raises:
        ValueError: If the required columns ("home_goals", "away_goals") are missing.
    """
    if "home_goals" not in goals.columns or "away_goals" not in goals.columns:
        raise ValueError(
            "Input DataFrame must contain 'home_goals' and 'away_goals' columns."
        )

    # Extract data from tuples
    home_id = goals["home_goals"].apply(lambda x: x[0])
    away_id = goals["home_goals"].apply(lambda x: x[1])
    home_goals = goals["home_goals"].apply(lambda x: x[2])
    away_goals = goals["away_goals"].apply(lambda x: x[2])

    # Calculate match outcome (toto)
    conditions = [
        (home_goals == away_goals),
        (home_goals > away_goals),
        (home_goals < away_goals),
    ]
    choices = [0, 1, 2]
    toto = np.select(conditions, choices)

    # Create vectorized DataFrame
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
