"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""
import typing as t

import arviz as az
import netCDF4
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from IPython.core.pylabtools import figsize
from kedro_datasets_experimental.netcdf import NetCDFDataset
from matplotlib import pyplot as plt
from modelbuilder import FootballModel, FootballModel_2
from utils import get_diffs, get_goal_distribution, get_probs_winner, get_teams

# nb_teams = len(teams)
# model1 = pm.Model()
min_mu = 0.0001
# average_goals = 3.0

low = 10e-8  # Constant


def init_model(team_lexicon, parameters: t.Dict, **kwargs) -> FootballModel:
    model = parameters["model"]

    if model == "f1":
        return FootballModel(
            model_config=parameters["model_config"],
            sampler_config=parameters["sampler_config"],
            team_lexicon=team_lexicon,
        )
    if model == "f2":
        try:
            kwargs["toto"]
        except KeyError:
            raise ValueError("f2 model requires toto as keyword argument.")

        return FootballModel_2(
            model_config=parameters["model_config"],
            sampler_config=parameters["sampler_config"],
            team_lexicon=team_lexicon,
            toto=kwargs["toto"],
        )
    else:
        raise ValueError(f"Model {model} not supported.")


def fit(
    model, x_data, y_data: pd.DataFrame, team_lexicon, parameters: t.Dict
) -> xr.Dataset:
    # model = init_model(parameters["model_config"]["model"], team_lexicon, parameters)
    goals = y_data.apply(lambda row: (row["home_goals"], row["away_goals"]), axis=1)

    idata = model.fit(
        X=x_data,
        y=goals,
        random_seed=parameters["random_seed"],
    )
    # idata = NetCDFDataset(
    #     filepath="data/06_models/model_1_idata_active.nc",
    #     load_args=dict(decode_times=False),
    # ).load()
    return az.convert_to_dataset(idata)


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


def plot_offence_defence(offence, defence, team_lexicon, **kwargs):
    # kwargs dict for histograms
    plot_hist_param = {
        "facecolor": "#2ab0ff",
        "edgecolor": "#169acf",
        "linewidth": 0.5,
        "density": True,
    }

    show_teams = 3
    # show_teams = nb_teams

    bins = 40
    fig, axes = plt.subplots(
        nrows=show_teams + 1, ncols=2, figsize=(10, (show_teams + 1) * 2)
    )

    for teamname, index_value in team_lexicon.iterrows():
        i = index_value[0]
        title = "Offence of " + teamname

        axes[i, 0].set_title(title)
        axes[i, 0].hist(offence[i], bins=bins, range=(0, 4.2), **plot_hist_param)
        title = "Defence of " + teamname
        axes[i, 1].set_title(title)
        axes[i, 1].hist(defence[i], bins=bins, range=(-2.0, 2.2), **plot_hist_param)

        if i >= show_teams:
            break

    fig.suptitle("Offence and defence distribution of the clubs.")
    fig.subplots_adjust(hspace=0.5)
    fig.tight_layout()
    return fig


def plot_goal_diffs(offence, defence, team_lexicon, parameters: t.Dict):
    plot_params = parameters["plot_params"]
    max_goals = plot_params["max_goals"]
    team_1 = team_lexicon.loc[plot_params["team_1"], "index"]
    team_2 = team_lexicon.loc[plot_params["team_2"], "index"]

    diff_ij, diff_ji = get_diffs(team_1, team_2, offence, defence)
    poisson_goals_1 = get_goal_distribution(diff_ij, max_goals)
    poisson_goals_2 = get_goal_distribution(diff_ji, max_goals)
    # Create the figure and axes objects, specify the size and the dots per inches
    fig, ax = plt.subplots(figsize=(6, 4), dpi=96)
    # Plot bars
    x = np.arange(0, max_goals)
    width = 0.4
    bar1 = ax.bar(
        x - width / 2,
        poisson_goals_1,
        width=width,
        alpha=0.9,
        label="Tore von " + plot_params["team_1"],
    )  # Add in title and subtitle
    bar2 = ax.bar(
        x + width / 2,
        poisson_goals_2,
        width=width,
        alpha=0.9,
        label="Tore von " + plot_params["team_2"],
    )  # Add in title and subtitle
    ax.set_xticks(x, x)
    ax.text(
        x=0.12,
        y=0.93,
        s="Torvorhersage",
        transform=fig.transFigure,
        ha="left",
        fontsize=10,
        weight="bold",
        alpha=1.0,
    )
    ax.text(
        x=0.12,
        y=0.90,
        s=plot_params["team_1"] + " gegen " + plot_params["team_2"],
        transform=fig.transFigure,
        ha="left",
        fontsize=8,
        alpha=1.0,
    )
    ax.legend()
    return fig


# figure for book
# plt.savefig('pics/figure_8    defence_dim_0  int64 1 (8000,).png')
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
