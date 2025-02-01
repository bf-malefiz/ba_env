from collections import OrderedDict

import numpy as np
import pandas as pd
from bundesliga.utils.validation import validate_dataframe

GOALS_HOME = "FTHG"
GOALS_AWAY = "FTAG"


@validate_dataframe(
    df_arg_name="df", required_columns=["HomeTeam", "AwayTeam"], allow_empty=False
)
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

    # Combine home and away teams and remove duplicates
    all_teams = (
        pd.concat([df["HomeTeam"], df["AwayTeam"]])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    team_indices = enumerate(all_teams)
    lex = pd.DataFrame(team_indices, columns=["index", "team"]).set_index("team")
    return lex


@validate_dataframe(
    df_arg_name="df",
    required_columns=["HomeTeam", "AwayTeam", GOALS_HOME, GOALS_AWAY],
    allow_empty=False,
)
@validate_dataframe(
    df_arg_name="lexicon",
    required_columns=["index"],
    required_index="team",
    allow_empty=False,
)
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

    try:
        df[GOALS_HOME] = pd.to_numeric(df[GOALS_HOME], errors="raise")
        df[GOALS_AWAY] = pd.to_numeric(df[GOALS_AWAY], errors="raise")
    except ValueError:
        raise ValueError(
            f"Non-numerical values detected in columns {GOALS_HOME} or {GOALS_AWAY}."
        )

    if (df["HomeTeam"] == df["AwayTeam"]).any():
        raise ValueError("HomeTeam and AwayTeam must not be the same (invalid match).")

    if (df[GOALS_HOME] < 0).any() or (df[GOALS_AWAY] < 0).any():
        raise ValueError(
            f"Negative goals detected in columns {GOALS_HOME}/{GOALS_AWAY}."
        )

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


@validate_dataframe(
    df_arg_name="goals",
    required_columns=["home_goals", "away_goals"],
    allow_empty=False,
)
def vectorize_data(goals: pd.DataFrame) -> pd.DataFrame:
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
