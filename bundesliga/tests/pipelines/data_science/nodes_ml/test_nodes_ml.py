"""
===============================================================================
TEST DOCUMENTATION: Bundesliga Data Science Nodes
===============================================================================
Title:
    Bundesliga Data Science Pipeline Unit Tests

Purpose:
    This test suite validates the functionality of key components in the
    Bundesliga machine learning pipeline. It includes tests for initializing
    various models (PyMC, Pyro), training them on sample data, generating
    predictions, evaluating results, and determining match outcomes.

Scope:
    - **Model Initialization**:
        Ensures different models (simple PyMC, simple Pyro, etc.) correctly instantiate.
    - **Training**:
        Confirms that training proceeds without errors and that required columns exist in the dataset.
    - **Predictions**:
        Validates that models produce predicted goal counts as expected.
    - **Evaluation**:
        Checks the computation of accuracy metrics and verifies that win/loss/tie outcomes are determined accurately.
    - **Edge Cases**:
        Tests for improper configurations, missing data columns, and unimplemented features.

Referenced Components:
    - **init_model**:
        Initializes and returns an appropriate model instance (PyMC or Pyro) based on user-defined options.
    - **train**:
        Invokes the model's training routine.
    - **predict_goals**:
        Calls the model to generate goal predictions.
    - **evaluate_match**:
        Computes performance metrics such as accuracy.
    - **determine_winner**:
        Determines match outcomes ("home", "away", "tie") based on predicted goal tallies.
    - **true_result**:
        Identifies the actual match outcome given the real final score.
    - **predicted_result**:
        Interprets probability distributions to label the predicted outcome as "home", "away", or "tie".

Testing Standards and Guidelines:
    1. **Structured Naming**:
        Test functions include the tested function’s name (e.g., `test_init_model_pymc_simple`) and a short scenario descriptor.
    2. **Fixtures**:
        - `model_options()`: Default model configuration.
        - `lexicon_expected()`: A sample lexicon mapping teams to indices.
        - `sample_train_data()`: Minimal training dataset.
        - `sample_test_data()`: Minimal testing dataset.
        - `dummy_predictions()`: Synthetic predictions for evaluation.
        These fixtures keep the test setup organized and reusable.
    3. **Error Handling**:
        Explicit checks for missing columns or invalid configurations that should raise specific exceptions (e.g., `ValueError`,`NotImplementedError`).
    4. **Clarity and Maintainability**:
        The test suite is divided into logical sections—initialization, training, prediction, evaluation, edge cases—promoting ease of future additions.

Pre-requisites:
    - Python 3.x
    - `pytest`, `pandas`, `numpy`
    - Mocking tools (e.g., `unittest.mock`) for simulating model methods

How to Run:
    1. Install the required dependencies (pandas, numpy, pytest).
    2. Navigate to the directory containing this file.
    3. Execute `pytest -v` in the terminal to run all tests.

Expected Outcome:
    - All tests should pass if the pipeline components and models are implemented correctly.
    - Error and failure messages will pinpoint any potential issues in model initialization, training, or evaluation.

===============================================================================
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from bundesliga.model.pymc.pymc_simple_model import SimplePymcModel
from bundesliga.model.pyro.pyro_simple_model import SimplePyroModel
from bundesliga.pipelines.data_science.nodes_ml.evaluate import (
    determine_winner,
    evaluate_match,
    predicted_result,
    true_result,
)
from bundesliga.pipelines.data_science.nodes_ml.train_model import (
    init_model,
    predict_goals,
    train,
)


@pytest.fixture
def model_options():
    """Provides default model configuration with empty model and engine settings."""
    return {
        "model": "",
        "engine": "",
        "start_match": 20,
        "walk_forward": 0,
        "test_size": 0.2,
        "model_config": {"prior_params": {}},
        "sampler_config": {},
    }


@pytest.fixture
def lexicon_expected():
    """Provides a sample lexicon mapping team names to numerical indices."""
    data = {"team": ["TeamA", "TeamB", "TeamC"], "index": [0, 1, 2]}
    lex = pd.DataFrame(data).set_index("team")
    return lex


@pytest.fixture
def sample_train_data():
    """Provides a minimal dataset for model training."""
    data = {
        "home_id": [0, 1],
        "away_id": [1, 0],
        "home_goals": [2, 1],
        "away_goals": [1, 2],
        "toto": [1, 2],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_test_data():
    """Provides a minimal dataset for model testing."""
    data = {
        "home_id": [0],
        "away_id": [1],
        "home_goals": [3],
        "away_goals": [2],
        "toto": [1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def dummy_predictions():
    """Provides dummy prediction results for evaluation tests."""
    return pd.DataFrame({"home_goals": [3], "away_goals": [2]})


def test_init_model_pymc_simple(lexicon_expected, model_options):
    """
    Tests if the PyMC simple model initializes correctly.

    Verifies that when model_options specify "simple" for the model and "pymc" for the engine,
    init_model returns an instance of SimplePymcModel.
    """
    model_options["model"] = "simple"
    model_options["engine"] = "pymc"
    model = init_model(lexicon_expected, model_options)
    assert isinstance(model, SimplePymcModel)


def test_init_model_pyro_simple(lexicon_expected, model_options):
    """
    Tests if the Pyro simple model initializes correctly.

    Verifies that when model_options specify "simple" for the model and "pyro" for the engine,
    init_model returns an instance of SimplePyroModel.
    """
    model_options["model"] = "simple"
    model_options["engine"] = "pyro"
    model = init_model(lexicon_expected, model_options)
    assert isinstance(model, SimplePyroModel)


def test_init_model_pyro_toto_not_implemented(lexicon_expected, model_options):
    """
    Tests that initializing a Pyro 'toto' model raises NotImplementedError.

    When model_options specify "toto" for the model and "pyro" for the engine,
    verifies that init_model raises a NotImplementedError with an appropriate message.
    """
    model_options["model"] = "toto"
    model_options["engine"] = "pyro"
    with pytest.raises(NotImplementedError, match="Pyro-toto not implemented."):
        init_model(lexicon_expected, model_options)


def test_train_happy_path(sample_train_data):
    """
    Ensures that the train function executes successfully with valid input data.

    Mocks a model's train method and confirms that train is called correctly,
    returning the model instance without errors.
    """
    mock_model = MagicMock()
    mock_model.train.return_value = "some_inference_data"
    parameters = {"lr": 0.01}
    returned_model = train(mock_model, sample_train_data, parameters)
    mock_model.train.assert_called_once()
    assert returned_model == mock_model


def test_train_missing_columns():
    """
    Ensures that the train function raises a ValueError when required columns are missing.

    Provides a DataFrame missing necessary columns and verifies that a ValueError
    is raised with an appropriate error message.
    """
    df_missing = pd.DataFrame({"home_id": [0]})
    mock_model = MagicMock()
    with pytest.raises(ValueError, match="Missing required columns"):
        train(mock_model, df_missing, {})


def test_predict_goals(sample_test_data):
    """
    Tests that the predict_goals function correctly invokes the model's prediction method.

    Mocks a model's predict_goals method, checks that it is called with the provided test data,
    and verifies that the output is a DataFrame.
    """
    mock_model = MagicMock()
    mock_model.predict_goals.return_value = pd.DataFrame(
        {"home_goals": [2], "away_goals": [1]}
    )
    pp = predict_goals(mock_model, sample_test_data, parameters={})
    mock_model.predict_goals.assert_called_once_with(sample_test_data)
    assert isinstance(pp, pd.DataFrame)


def test_evaluate_home_win(sample_test_data, dummy_predictions):
    """
    Tests the evaluate_match function for a scenario where the home team wins.

    Mocks the model's predict_toto_probabilities method to return probabilities favoring a home win,
    and verifies that evaluate_match interprets the prediction correctly as "home".
    """
    mock_model = MagicMock()
    mock_model.predict_toto_probabilities.return_value = np.array([[0.6, 0.2, 0.2]])
    results = evaluate_match(
        mock_model, match=1, test_data=sample_test_data, predictions=dummy_predictions
    )
    assert results["predicted_result"] == "home"


def test_determine_winner():
    """
    Tests the determine_winner function for correctly assigning match outcomes.

    Provides a DataFrame with varying home and away goal counts and verifies that the resulting
    'winner' column correctly indicates "home", "away", or "tie" for each row.
    """
    preds = pd.DataFrame({"home_goals": [2, 1, 3], "away_goals": [1, 2, 3]})
    result = determine_winner(preds)
    assert "winner" in result.columns
    assert result.loc[0, "winner"] == "home"
    assert result.loc[1, "winner"] == "away"
    assert result.loc[2, "winner"] == "tie"


def test_true_result():
    """
    Verifies that the true_result function correctly determines the actual match outcome.

    Checks that given sample goal data for home win, away win, and tie scenarios,
    true_result returns "home", "away", and "tie" respectively.
    """
    row_home_win = pd.Series({"home_goals": 3, "away_goals": 1})
    row_away_win = pd.Series({"home_goals": 1, "away_goals": 4})
    row_tie = pd.Series({"home_goals": 2, "away_goals": 2})
    assert true_result(row_home_win) == "home"
    assert true_result(row_away_win) == "away"
    assert true_result(row_tie) == "tie"


def test_predicted_result():
    """
    Verifies that the predicted_result function returns the outcome with the highest probability.

    Given sample probability distributions for different scenarios,
    confirms that predicted_result identifies "home", "away", or "tie" appropriately.
    """
    probs_home = pd.Series({"home": 0.5, "away": 0.3, "tie": 0.2})
    probs_away = pd.Series({"home": 0.3, "away": 0.6, "tie": 0.1})
    probs_tie = pd.Series({"home": 0.2, "away": 0.2, "tie": 0.6})
    assert predicted_result(probs_home) == "home"
    assert predicted_result(probs_away) == "away"
    assert predicted_result(probs_tie) == "tie"
