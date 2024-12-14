"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from kedro_datasets_experimental.netcdf import NetCDFDataset

"""PYMC Model 1
    """


# nb_teams = len(teams)
# model1 = pm.Model()
# min_mu = 0.0001
# average_goals = 3.0


def model_1(model_input_table: pd.DataFrame) -> None:
    nb_teams = 18
    nb_samples = 500
    tune = nb_samples // 10
    min_mu = 0.0001
    average_goals = 3.0  # average nb of goals in a game

    with pm.Model() as model:
        offence = pm.Normal("offence", tau=1.0, mu=average_goals / 2, shape=nb_teams)
        defence = pm.Normal("defence", tau=1.0, mu=0.0, shape=nb_teams)

        offence_home = offence[model_input_table["home_id"]]
        defence_home = defence[model_input_table["home_id"]]
        offence_away = offence[model_input_table["away_id"]]
        defence_away = defence[model_input_table["away_id"]]

        mu_home = offence_home - defence_away
        mu_away = offence_away - defence_home
        # note: use exponent in practice instead of switch
        mu_home = pm.math.switch(mu_home > min_mu, mu_home, min_mu)
        mu_away = pm.math.switch(mu_away > min_mu, mu_away, min_mu)

        pm.Poisson("home_goals", observed=model_input_table["home_goals"], mu=mu_home)
        pm.Poisson("away_goals", observed=model_input_table["away_goals"], mu=mu_away)
        trace = pm.sample(draws=nb_samples, tune=tune)
        posterior = trace.posterior.stack(sample=["chain", "draw"])
        offence = posterior["offence"]
        defence = posterior["defence"]
        # home_advantage = posterior["home_advantage"]

        # pm.predictions_to_inference_data

        samples = pm.sample_posterior_predictive(trace)


def test():
    data = np.random.rand(4, 3)

    locs = ["IA", "IL", "IN"]

    times = pd.date_range("2000-01-01", periods=4)

    foo = xr.DataArray(data, coords=[times, locs], dims=["time", "space"])
    return foo
