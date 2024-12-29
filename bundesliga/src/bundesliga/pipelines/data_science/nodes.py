"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""
import typing as t

import arviz as az
import netCDF4
import pandas as pd
import pymc as pm
from kedro.io import DataCatalog
from kedro_datasets_experimental.netcdf import NetCDFDataset
from model_interfaces import pymc_FootballModel
from modelbuilder import FootballModel

"""PYMC Model 1
    """


# nb_teams = len(teams)
# model1 = pm.Model()
# min_mu = 0.0001
# average_goals = 3.0
def _init_model(model: str, parameters: t.Dict, **kwargs):
    if model == "pymc":
        return FootballModel(
            model_config=parameters["model_config"],
            sampler_config=parameters["sampler_config"],
        )
    else:
        raise ValueError(f"Model {model} not supported.")


def fit(
    x_data, y_data: pd.DataFrame, parameters: t.Dict, model="pymc"
) -> netCDF4.Dataset:
    model = _init_model(model, parameters)
    goals = y_data.apply(lambda row: (row["home_goals"], row["away_goals"]), axis=1)

    idata = model.fit(
        X=x_data,
        y=goals,
        random_seed=parameters["random_seed"],
    )

    # posterior = idata.posterior.stack(sample=["chain", "draw"])
    # offence = posterior["offence"]
    # defence = posterior["defence"]
    # # home_advantage = posterior["home_advantage"]
    # # pm.predictions_to_inference_data

    # samples = pm.sample_posterior_predictive(trace)

    # idata = NetCDFDataset(
    #     filepath="data/06_models/model_1_idata_active.nc",
    #     load_args=dict(decode_times=False),
    # ).load()

    # toDO:
    # - was wollen wir predicten?
    # - gibt es eine moeglichkeit home+away goals zu predicten? es wird nur eine Dimesnsion erlaubt (var_output is name der im posterior_predictive vorkommen muss)
    # - wie predicten wir ein spiel zu einem bestimmten zeitpunkt?

    # result = pd.DataFrame(0, index=x_data.index, columns=x_data.columns)
    # result.iloc[9] = x_data.iloc[9]
    # samples = model.predict(result)

    # return az.convert_to_dataset(idata), az.convert_to_dataset(samples)
    return az.convert_to_dataset(idata)
