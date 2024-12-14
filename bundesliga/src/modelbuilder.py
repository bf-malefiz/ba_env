from typing import Dict, List, Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from numpy.random import RandomState
from pymc_experimental.model_builder import ModelBuilder

RANDOM_SEED = 8927
min_mu = 0.0001
average_goals = 3.0

rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")


class FootballModel(ModelBuilder):
    # def __init__(self, model_config, model_coords):
    #     self.model_config = model_config
    #     self.model_coords = model_coords
    #     self.model = None

    # Give the model a name
    _model_type = "FootballModel"

    # And a version
    version = "0.1"

    def build_model(self, X: pd.DataFrame, goals: pd.Series, **kwargs):
        """
        build_model creates the PyMC model

        Parameters:
        model_config: dictionary
            it is a dictionary with all the parameters that we need in our model example:  a_loc, a_scale, b_loc
        X : pd.DataFrame
            The input data that is going to be used in the model. This should be a DataFrame
            containing the features (predictors) for the model. For efficiency reasons, it should
            only contain the necessary data columns, not the entire available dataset, as this
            will be encoded into the data used to recreate the model.

        y : pd.Series
            The target data for the model. This should be a Series representing the output
            or dependent variable for the model.

        kwargs : dict
            Additional keyword arguments that may be used for model configuration.
        """
        # Check the type of X and y and adjust access accordingly
        X_values = X[["home_id", "away_id"]].values
        goals = goals.values if isinstance(goals, pd.Series) else goals

        self._generate_and_preprocess_model_data(X_values, goals)

        with pm.Model(coords=self.model_coords) as self.model:
            # Create mutable data containers
            x_data = pm.Data("x_data", X_values)
            home_goals_data = pm.Data(
                "home_goals_data", goals[:, 0]
            )  # check tuple unpacking
            away_goals_data = pm.Data("away_goals_data", goals[:, 1])

            # prior parameters
            shape_priors = self.model_config.get("shape_priors", 1)
            # offence_mu_prior = avg_goals/2    3/2
            offence_mu_prior = self.model_config.get(
                "offence_mu_prior", average_goals / 2
            )
            offence_tau_prior = self.model_config.get("offence_tau_prior", 1.0)
            defence_mu_prior = self.model_config.get("defence_mu_prior", 0.0)
            defence_tau_prior = self.model_config.get("defence_tau_prior", 1.0)

            # priors
            offence = pm.Normal(
                "offence",
                tau=offence_tau_prior,
                mu=offence_mu_prior,
                shape=shape_priors,
            )
            defence = pm.Normal(
                "defence",
                tau=defence_tau_prior,
                mu=defence_mu_prior,
                shape=shape_priors,
            )

            offence_home = offence[x_data[:, 0]]
            defence_home = defence[x_data[:, 0]]
            offence_away = offence[x_data[:, 1]]
            defence_away = defence[x_data[:, 1]]

            mu_home = offence_home - defence_away
            mu_away = offence_away - defence_home
            # note: use exponent in practice instead of switch
            mu_home = pm.math.switch(mu_home > min_mu, mu_home, min_mu)
            mu_away = pm.math.switch(mu_away > min_mu, mu_away, min_mu)

            # observed
            pm.Poisson("home_goals", observed=goals[0], mu=mu_home)
            pm.Poisson("away_goals", observed=goals[1], mu=mu_away)

            # # with model1_home_advantage:
            # trace = pm.sample(draws=nb_samples, tune=tune)
            # posterior = trace.posterior.stack(sample=["chain", "draw"])
            # offence = posterior["offence"]
            # defence = posterior["defence"]
            # # home_advantage = posterior["home_advantage"]
            # # pm.predictions_to_inference_data

            # samples = pm.sample_posterior_predictive(trace)

    def _data_setter(
        self,
        X: Union[pd.DataFrame, pd.Series],
        home_goals: Union[pd.Series, np.ndarray],
        away_goals: Union[pd.Series, np.ndarray],
    ) -> None:
        if isinstance(X, pd.DataFrame):
            x_values = X[["home_id", "away_id"]].values
        else:
            # Assuming "input" is the first column
            x_values = X[:, 0]

        with self.model:
            pm.set_data({"x_data": x_values})
            if home_goals is not None:
                pm.set_data(
                    {
                        "home_goals": home_goals.values
                        if isinstance(home_goals, pd.Series)
                        else home_goals
                    }
                )
            if away_goals is not None:
                pm.set_data(
                    {
                        "away_goals": away_goals.values
                        if isinstance(away_goals, pd.Series)
                        else away_goals
                    }
                )

    @staticmethod
    def get_default_model_config() -> dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization.
        The model config dict is generally used to specify the prior values we want to build the model with.
        It supports more complex data structures like lists, dictionaries, etc.
        It will be passed to the class instance on initialization, in case the user doesn't provide any model_config of their own.
        """
        model_config: dict = {
            "offence_mu_prior": 0.0,
            "offence_tau_prior": 1.0,
            "defence_mu_prior": 0.0,
            "defence_tau_prior": 1.0,
            "shape_priors": 18,  # toDo: check this for the number of teams
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.
        """
        sampler_config: dict = {
            "draws": 1_000,
            "tune": 1_00,  # 10% of draws
            "chains": 3,
            "target_accept": 0.95,
        }
        return sampler_config

    @property
    def output_var(self):
        return "y"

    @property
    def _serializable_model_config(self) -> dict[str, Union[int, float, dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config

    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.

        These parameters are stored within `idata.attrs` using keys that correspond to the parameter names.
        If you don't need to store any extra parameters, you can leave this method unimplemented.

        Example:
            For saving customer IDs provided as an 'customer_ids' input to the model:
            self.customer_ids = customer_ids.values #this line is done outside of the function, preferably at the initialization of the model object.
            idata.attrs["customer_ids"] = json.dumps(self.customer_ids.tolist())  # Convert numpy array to a JSON-serializable list.
        """
        pass

        pass

    def _generate_and_preprocess_model_data(
        self,
        X: Union[pd.DataFrame, pd.Series],
        home_goals: Union[pd.Series, np.ndarray],
        away_goals: Union[pd.Series, np.ndarray],
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """
        self.model_coords = {
            "team": teams,
            "match": np.arange(60),
            "field": ["home", "away"],
        }

        self.X_values = X
        self.home_goals = home_goals
        self.away_goals = away_goals
