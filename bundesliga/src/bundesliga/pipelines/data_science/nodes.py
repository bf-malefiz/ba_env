"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""
import arviz as az
import netCDF4
import pandas as pd
import xarray as xr

# from kedro_datasets_experimental.netcdf import NetCDFDataset
from modelbuilder import FootballModel

"""PYMC Model 1
    """


# nb_teams = len(teams)
# model1 = pm.Model()
# min_mu = 0.0001
# average_goals = 3.0


def model_1(model_input_table: pd.DataFrame) -> netCDF4.Dataset:
    x_input = model_input_table[["home_id", "away_id"]]
    goals = model_input_table[["home_goals", "away_goals"]].apply(
        lambda row: (row["home_goals"], row["away_goals"]), axis=1
    )

    model = FootballModel()
    model_1_idata = model.fit(
        X=x_input,
        y=goals,
    )
    return az.convert_to_dataset(model_1_idata)
