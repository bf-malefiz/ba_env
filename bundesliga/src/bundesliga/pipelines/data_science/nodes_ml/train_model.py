import typing as t

import arviz as az
import pandas as pd
import pymc as pm
import xarray as xr
from bundesliga.model.modelbuilder import FootballModel, FootballModel_2


def split_time_data(x_data: pd.DataFrame, y_data: pd.DataFrame, current_day: int):
    """
    Gibt train_X, train_y, test_X, test_y zurück,
    basierend auf der Spalte 'matchday' in X / y.
    """
    x_data = x_data[:current_day]
    y_data = x_data[: current_day + 1]

    return x_data, y_data


def init_model(parameters: t.Dict, **kwargs) -> xr.Dataset:
    model = parameters["model"]

    if model == "f1":
        return FootballModel()
    if model == "f2":
        try:
            kwargs["toto"]
        except KeyError:
            raise ValueError("f2 model requires toto as keyword argument.")

        return FootballModel_2(
            toto=kwargs["toto"],
        )
    else:
        raise ValueError(f"Model {model} not supported.")


def fit(
    x_data, y_data: pd.DataFrame, model, parameters: t.Dict, **kwargs
) -> xr.Dataset:
    goals = y_data.apply(lambda row: (row["home_goals"], row["away_goals"]), axis=1)
    idata = model.fit(
        X=x_data,
        y=goals,
        random_seed=parameters["random_seed"],
    )

    return model, idata


def predict(model, idata, x_data, parameters):
    with model.model:
        pred_home_away_mean = model.predict_both_goals(
            x_data, idata, extend_idata=False
        )

    return pred_home_away_mean
