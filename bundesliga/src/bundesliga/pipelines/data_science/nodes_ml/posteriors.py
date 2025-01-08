import arviz as az
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from kedro_datasets_experimental.netcdf import NetCDFDataset


def posterior_f1(idata: netCDF4.Dataset) -> xr.Dataset:
    idata = az.convert_to_inference_data(idata)
    posterior = idata.posterior.stack(sample=["chain", "draw"])

    offence = posterior["offence"]
    defence = posterior["defence"]

    return offence.reset_index("sample"), defence.reset_index("sample")


def posterior_f2(idata: netCDF4.Dataset) -> xr.Dataset:
    idata = az.convert_to_inference_data(idata)
    posterior = idata.posterior.stack(sample=["chain", "draw"])

    weights = posterior["weights"]
    offence_defence_diff = posterior["offence_defence_diff"]
    score = posterior["score"]
    home_advantage = posterior["home_advantage"]

    return (
        weights.reset_index("sample"),
        offence_defence_diff.reset_index("sample"),
        score.reset_index("sample"),
        home_advantage.reset_index("sample"),
    )


def team_means(idata: netCDF4.Dataset) -> xr.Dataset:
    idata = az.convert_to_dataset(idata)
    means = idata.mean(dim=["sample"])

    return means
