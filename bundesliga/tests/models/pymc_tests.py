import pandas as pd
import pytest

from bundesliga.model.pymc.pymc_simple_model import SimplePymcModel


def test_predict_goals():
    model = SimplePymcModel()
    test_data = pd.DataFrame({"home_id": [1, 2], "away_id": [2, 1]})
    predictions = model.predict_goals(test_data)
    assert isinstance(predictions, pd.DataFrame)
    assert "home_goals" in predictions.columns
    assert "away_goals" in predictions.columns
