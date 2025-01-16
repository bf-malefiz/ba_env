import typing as t

import arviz as az
import bundesliga.model.pymc.pymc_simple_model as pm_simple
import bundesliga.model.pymc.pymc_toto_model as pm_toto
import bundesliga.model.pyro.pyro_simple_model as pyro_simple
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import xarray as xr


def init_model(team_lexicon, parameters: t.Dict, **kwargs) -> xr.Dataset:
    model = parameters["model"]
    engine = parameters["engine"]

    match engine:
        case "pymc":
            match model:
                case "simple":
                    return pm_simple.SimpleModel(parameters)
                case "toto":
                    try:
                        kwargs["toto"]
                    except KeyError:
                        raise ValueError(
                            "pm_toto.TotoModel requires 'toto' as keyword argument."
                        )
                    return pm_toto.TotoModel(toto=kwargs["toto"])
        case "pyro":
            match model:
                case "simple":
                    return pyro_simple.SimplePyroModel(team_lexicon, parameters)
                case "toto":
                    raise NotImplementedError("Pyro not implemented.")
        case "stan":
            match model:
                case "simple":
                    raise NotImplementedError("Stan not implemented.")
                case "toto":
                    raise NotImplementedError("Stan not implemented.")
        case _:
            raise ValueError(f"Model {model} not supported.")


def train(model, train_data, parameters: t.Dict, **kwargs) -> xr.Dataset:
    X = train_data[["home_id", "away_id"]]
    y = train_data[["home_goals", "away_goals", "toto"]]

    idata = model.train(X=X, y=y, parameters=parameters)

    return model, idata


def predict(model, test_data, parameters, **kwargs) -> pd.DataFrame:
    pp = model.predict_toto_probabilities(test_data)

    return pd.DataFrame(pp, columns=["tie", "home", "away"])


def evaluate(model, day, vectorized_data, predictions):
    az.style.use("arviz-doc")
    az.load_arviz_data("non_centered_eight")

    # a=az.plot_trace(idata)
    # az.plot_ppc(
    #     idata,
    #     data_pairs={"match": "home_goals"},
    #     alpha=0.03,
    #     textsize=14,
    #     num_pp_samples=1000,
    #     random_seed=42,
    # )
    # az.plot_ppc(
    #     idata,
    #     alpha=0.03,
    #     textsize=14,
    #     num_pp_samples=10,
    #     random_seed=42,
    # )
    # # az.plot_ppc(data, data_pairs={"match": "match"}, alpha=0.03, textsize=14)

    # plt.show()
    print("s")
    return pd.DataFrame([[1], [1]], columns=["s"])
