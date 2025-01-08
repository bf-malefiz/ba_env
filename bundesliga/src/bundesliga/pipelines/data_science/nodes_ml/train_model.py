import typing as t

import arviz as az
import pandas as pd
import xarray as xr
from modelbuilder import FootballModel, FootballModel_2

# def create_training_data(features, labels, test_size, random_state):
#     x_train, x_test, y_train, y_test = train_test_split(
#         features, labels, test_size=test_size, random_state=random_state
#     )
#     return x_train, x_test, y_train, y_test


def fit(
    x_data, y_data: pd.DataFrame, team_lexicon, parameters: t.Dict, **kwargs
) -> xr.Dataset:
    def _init_model() -> FootballModel:
        model = parameters["model"]

        if model == "f1":
            return FootballModel(
                # model_config=parameters["model_config"],
                # sampler_config=parameters["sampler_config"],
                # team_lexicon=team_lexicon,
            )
        if model == "f2":
            try:
                kwargs["toto"]
            except KeyError:
                raise ValueError("f2 model requires toto as keyword argument.")

            return FootballModel_2(
                # model_config=parameters["model_config"],
                # sampler_config=parameters["sampler_config"],
                # team_lexicon=team_lexicon,
                toto=kwargs["toto"],
            )
        else:
            raise ValueError(f"Model {model} not supported.")

    goals = y_data.apply(lambda row: (row["home_goals"], row["away_goals"]), axis=1)

    model = _init_model()
    idata = model.fit(
        X=x_data,
        y=goals,
        random_seed=parameters["random_seed"],
    )
    ds = az.convert_to_dataset(idata)
    # idata = NetCDFDataset(
    #     filepath="data/06_models/model_1_idata_active.nc",
    #     load_args=dict(decode_times=False),
    # ).load()
    s = az.convert_to_dataset(model.idata)
    sidata = model.sample_posterior_predictive(
        x_data, extend_idata=True, combined=True, var_names=["home_goals", "away_goals"]
    )

    return az.convert_to_dataset(idata)
