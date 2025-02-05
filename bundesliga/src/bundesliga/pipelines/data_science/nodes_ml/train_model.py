import typing as t

import xarray as xr

import bundesliga.model.pymc.pymc_simple_model as pm_simple
import bundesliga.model.pyro.pyro_simple_model as pyro_simple
from bundesliga.utils.validation import validate_dataframe


@validate_dataframe(
    df_arg_name="team_lexicon",
    required_columns=["index"],
    required_index="team",
    allow_empty=False,
)
def init_model(team_lexicon, model_options: t.Dict, **kwargs) -> xr.Dataset:
    model = model_options["model"]
    engine = model_options["engine"]

    match engine:
        case "pymc":
            match model:
                case "simple":
                    return pm_simple.SimplePymcModel(model_options, team_lexicon)
                case "toto":
                    try:
                        kwargs["toto"]
                    except KeyError:
                        raise ValueError(
                            "pm_toto.TotoModel requires 'toto' as keyword argument."
                        )
                    raise NotImplementedError("Pymc-toto not implemented.")
        case "pyro":
            match model:
                case "simple":
                    return pyro_simple.SimplePyroModel(team_lexicon, model_options)
                case "toto":
                    raise NotImplementedError("Pyro-toto not implemented.")

        case _:
            raise ValueError(f"Engine {engine} not supported.")


@validate_dataframe(
    df_arg_name="train_data",
    required_columns=["home_id", "away_id", "home_goals", "away_goals", "toto"],
    allow_empty=False,
)
def train(model, train_data, model_options: t.Dict, **kwargs) -> xr.Dataset:
    X = train_data[["home_id", "away_id"]]
    y = train_data[["home_goals", "away_goals", "toto"]]

    idata = model.train(X=X, y=y, parameters=model_options)

    return model


@validate_dataframe(
    df_arg_name="test_data",
    required_columns=["home_id", "away_id"],
    allow_empty=False,
)
def predict_goals(model, test_data, **kwargs):
    pp = model.predict_goals(test_data)

    return pp
