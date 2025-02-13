"""
===============================================================================
TEST DOCUMENTATION: Bundesliga Data Processing Nodes
===============================================================================
Title: Data Processing Unit Tests

Purpose:
    This test suite is designed to validate the core data-processing functions
    in the 'bundesliga.pipelines.data_processing.nodes_etl.nodes' module. The
    tests ensure that each node behaves correctly given valid and invalid
    inputs. The goal is to maintain data integrity and correctness across all
    stages of the ETL (Extract, Transform, Load) pipeline.

Scope:
    The tests focus primarily on:
        1. Building a team lexicon ('build_team_lexicon')
        2. Getting goal results ('get_goal_results')
        3. Vectorizing data ('vectorize_data')
        4. Handling edge cases (invalid or inconsistent data)

Referenced Components:
    - build_team_lexicon():
        Generates a lexicon (DataFrame) mapping team names to unique numeric indices.
    - get_goal_results():
        Extracts goal tuples and merges them with team indices.
    - vectorize_data():
        Converts goal result data into a vectorized format suitable for downstream modeling or analysis.

Testing Standards and Guidelines:
    1. **Readability and Naming**:
        Each test function name includes the name of the function under test (e.g., test_build_team_lexicon_*).
    2. **Fixtures**:
        - sample_data() supplies a representative set of match records.
        - lexicon_expected() supplies the expected output DataFrame for build_team_lexicon().
        - goals_df_expected() supplies the expected output for get_goal_results().
        These fixtures follow a consistent naming convention, are reusable, and serve as single points of truth for test data.
    3. **Edge Case Coverage**:
        Tests explicitly check for error handling in situations like empty DataFrames, missing columns, same home/away teams, negative goals, or non-numeric goals.
    4. **Assertions**:
        Each test uses `pytest` to assert expected behavior. Where errors are expected, test functions confirm that the correct error message is raised.
    5. **Maintainability**:
        Test code is separated into logical sections, ensuring clarity on which function is under test. Any new function additions should follow the same pattern.

Pre-requisites:
    - Python 3.x
    - pytest
    - pandas

How to Run:
    1. Install pytest via pip if not already installed.
    2. From the command line, navigate to the directory containing this file.
    3. Run `pytest -v` (verbosity recommended for detailed output).

Expected Outcome:
    - All tests should pass if the nodes are correctly implemented.
    - Failures or errors will help diagnose issues with data handling (e.g., missing columns, incorrect lexicon assignment, etc.).

===============================================================================
"""

# =============================================================================
#                               Fixtures
# =============================================================================
import pandas as pd
import pytest

from bundesliga.pipelines.data_processing.nodes_etl.nodes import (
    build_team_lexicon,
    get_goal_results,
    vectorize_data,
)


@pytest.fixture
def sample_data():
    """
    Provides sample match data in DataFrame format,
    simulating a real-world CSV input.
    """
    data = {
        "HomeTeam": ["TeamA", "TeamB", "TeamA", "TeamC"],
        "AwayTeam": ["TeamB", "TeamA", "TeamC", "TeamA"],
        "FTHG": [1, 2, 3, 0],  # Full-time home goals
        "FTAG": [0, 1, 1, 2],  # Full-time away goals
    }
    return pd.DataFrame(data)


@pytest.fixture
def lexicon_expected():
    """
    Provides the expected output of `build_team_lexicon(sample_data)`.
    The order of teams follows their first occurrence in `sample_data`.
    """
    data = {"team": ["TeamA", "TeamB", "TeamC"], "index": [0, 1, 2]}
    df = pd.DataFrame(data).set_index("team")
    return df


@pytest.fixture
def goals_df_expected():
    """
    Provides the expected output of `get_goal_results(sample_data, lexicon_expected)`.
    Each column contains tuples (home_id, away_id, goals).
    """
    data = {
        "home_goals": [(0, 1, 1), (1, 0, 2), (0, 2, 3), (2, 0, 0)],
        "away_goals": [(0, 1, 0), (1, 0, 1), (0, 2, 1), (2, 0, 2)],
    }
    return pd.DataFrame(data)


# =============================================================================
#                            Tests: build_team_lexicon
# =============================================================================
def test_build_team_lexicon_empty_df():
    """
    Ensures that build_team_lexicon raises an error for an empty DataFrame.

    This test verifies that when an empty DataFrame (with the required columns) is passed to
    build_team_lexicon, the function correctly raises a ValueError indicating that the DataFrame is empty.
    """
    empty_df = pd.DataFrame(columns=["HomeTeam", "AwayTeam"])

    with pytest.raises(ValueError) as excinfo:
        build_team_lexicon(empty_df)

    assert "DataFrame 'df' is empty" in str(excinfo.value)


def test_build_team_lexicon(sample_data):
    """
    Tests that build_team_lexicon correctly creates a lexicon mapping team names to numeric indices.

    The test checks that the output DataFrame has teams as its index, includes the 'index' column,
    and contains the expected number of teams.
    """
    lex = build_team_lexicon(sample_data)

    expected_teams = {"TeamA", "TeamB", "TeamC"}
    actual_teams = set(lex.index)
    assert actual_teams == expected_teams

    assert "index" in lex.columns, "The 'index' column is missing in the lexicon."
    assert len(lex) == 3, "Incorrect number of teams in the lexicon."


def test_build_team_lexicon_missing_columns():
    """
    Ensures that build_team_lexicon raises an error when required columns are missing.

    This test provides a DataFrame that lacks the necessary 'HomeTeam' and 'AwayTeam' columns and
    asserts that a ValueError is raised with an appropriate error message.
    """
    incomplete_df = pd.DataFrame(
        {"FTHG": [1], "FTAG": [2]}
    )  # Missing HomeTeam/AwayTeam

    with pytest.raises(ValueError) as excinfo:
        build_team_lexicon(incomplete_df)

    assert "Missing required columns" in str(excinfo.value)


# =============================================================================
#                          Tests: get_goal_results
# =============================================================================


def test_get_goal_results(sample_data, lexicon_expected):
    """
    Tests that get_goal_results correctly extracts goal results and assigns team indices.

    This test verifies that the function produces an output DataFrame containing the columns 'home_goals'
    and 'away_goals', where each entry is a tuple of (home_id, away_id, goals). It confirms that the structure
    of the output matches expectations.
    """
    goals = get_goal_results(sample_data, lexicon_expected)

    expected_cols = {"home_goals", "away_goals"}
    assert expected_cols.issubset(goals.columns)

    first_home_goals = goals["home_goals"].iloc[0]
    assert len(first_home_goals) == 3


def test_get_goal_results_missing_columns(lexicon_expected):
    """
    Ensures that get_goal_results raises an error when required columns are missing.

    The test uses a DataFrame missing the 'AwayTeam' and 'FTAG' columns and asserts that the function
    raises a ValueError indicating the missing required columns.
    """
    bad_df = pd.DataFrame(
        {"HomeTeam": ["TeamX"], "FTHG": [1]}
    )  # Missing 'AwayTeam' & 'FTAG'

    with pytest.raises(ValueError) as excinfo:
        get_goal_results(bad_df, lexicon_expected)

    assert "Missing required columns" in str(excinfo.value)


def test_get_goal_results_id_alignment(sample_data, lexicon_expected):
    """
    Verifies that the team indices in the goal results match those in the lexicon.

    For each match in the sample data, the test checks that the home and away team indices in the tuples
    produced by get_goal_results align with the corresponding indices in the expected lexicon.
    """
    goals = get_goal_results(sample_data, lexicon_expected)

    for i, row in sample_data.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        home_id_expected = lexicon_expected.loc[home_team, "index"]
        away_id_expected = lexicon_expected.loc[away_team, "index"]

        home_goals_tuple = goals["home_goals"].iloc[i]
        away_goals_tuple = goals["away_goals"].iloc[i]

        assert home_goals_tuple[0] == home_id_expected
        assert home_goals_tuple[1] == away_id_expected

        assert away_goals_tuple[0] == home_id_expected
        assert away_goals_tuple[1] == away_id_expected


# =============================================================================
#                          Tests: vectorize_data
# =============================================================================


def test_vectorize_data(goals_df_expected):
    """
    Tests that vectorize_data produces the expected output DataFrame.

    This test confirms that the output from vectorize_data includes the required columns
    ('home_id', 'away_id', 'home_goals', 'away_goals', 'toto') and that the number of rows matches
    the input goal DataFrame.
    """
    result = vectorize_data(goals_df_expected)

    expected_cols = {"home_id", "away_id", "home_goals", "away_goals", "toto"}
    assert expected_cols.issubset(result.columns)

    assert len(result) == len(goals_df_expected)


def test_vectorize_data_missing_columns():
    """
    Ensures that vectorize_data raises an error when required columns are missing.

    The test passes a DataFrame missing the expected goal columns to vectorize_data and asserts that
    a ValueError is raised with an appropriate error message.
    """
    bad_goals = pd.DataFrame({"some_column": [1]})

    with pytest.raises(ValueError) as excinfo:
        vectorize_data(bad_goals)

    assert "Missing required columns" in str(excinfo.value)


# =============================================================================
#                          Tests: Edge Cases
# =============================================================================


def test_same_home_away_team():
    """
    Ensures that get_goal_results raises an error when the HomeTeam is the same as the AwayTeam.

    This test provides a DataFrame where at least one match has identical values for HomeTeam and AwayTeam,
    and verifies that get_goal_results raises a ValueError indicating an invalid match.
    """
    df = pd.DataFrame(
        {
            "HomeTeam": ["TeamA", "TeamB", "TeamB"],
            "AwayTeam": ["TeamA", "TeamB", "TeamA"],
            "FTHG": [1, 2, 3],
            "FTAG": [1, 2, 0],
        }
    )

    lex = build_team_lexicon(df)

    with pytest.raises(ValueError) as excinfo:
        get_goal_results(df, lex)

    assert "invalid match" in str(excinfo.value)


def test_negative_goals():
    """
    Ensures that get_goal_results raises an error when negative goal values are encountered.

    The test supplies a DataFrame with negative values in the goal columns and asserts that a ValueError
    is raised due to the invalid negative goal values.
    """
    df = pd.DataFrame(
        {
            "HomeTeam": ["TeamA"],
            "AwayTeam": ["TeamB"],
            "FTHG": [-1],
            "FTAG": [2],
        }
    )

    lex = build_team_lexicon(df)

    with pytest.raises(ValueError):
        get_goal_results(df, lex)


def test_non_numeric_goals():
    """
    Ensures that get_goal_results raises an error when non-numeric goal values are provided.

    This test passes a DataFrame with non-numeric data in the goal column (e.g., a string) and checks that
    a ValueError is raised, indicating that non-numeric values are not acceptable.
    """
    df = pd.DataFrame(
        {
            "HomeTeam": ["TeamA"],
            "AwayTeam": ["TeamB"],
            "FTHG": ["n/a"],  # String instead of integer
            "FTAG": [2],
        }
    )

    lex = build_team_lexicon(df)

    with pytest.raises(ValueError):
        get_goal_results(df, lex)
