from typing import Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from numpy.random import RandomState
from pymc_experimental.model_builder import ModelBuilder

RANDOM_SEED = 8927
min_mu = 0.0001
average_goals = 3.0

rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")


class FootballModel(ModelBuilder):
    # def __init__(self, X):
    #     self.X = X
    #     self.y = None
    #     self.model_config = None
    #     self.model_coords = None
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

        # self._generate_and_preprocess_model_data(X, goals)

        with pm.Model(coords=self.model_coords) as self.model:
            # Create mutable data containers
            team_idx = pm.Data(
                "team_idx",
                self.X[["home_id", "away_id"]].values,
                dims=("match", "field"),
            )
            home_goals = pm.Data(
                "home_goals",
                self.y["home_goals"].values,
                dims="match",
            )
            away_goals = pm.Data(
                "away_goals",
                self.y["away_goals"].values,
                dims="match",
            )
            # prior parameters
            off_mu_prior = self.model_config.get("off_mu_prior", 3 / 2)
            def_mu_prior = self.model_config.get("def_mu_prior", 0.0)
            off_tau_prior = self.model_config.get("off_tau_prior", 1.0)
            def_tau_prior = self.model_config.get("def_tau_prior", 1.0)

            # priors
            offence = pm.Normal(
                "offence",
                mu=off_mu_prior,
                sigma=off_tau_prior,
                # shape=shape_priors, ### kann ich auf shapes verzichten, wenn dims verwendet werden?
                dims="team",
            )
            defence = pm.Normal(
                "defence",
                mu=def_mu_prior,
                tau=def_tau_prior,
                dims="team",
            )

            offence_home_away = offence[team_idx]
            defence_home_away = defence[team_idx]

            mu_home_away = offence_home_away - defence_home_away.eval()[:, [1, 0]]

            mu_home = mu_home_away[:, 0]
            mu_away = mu_home_away[:, 1]

            # # note: use exponent in practice instead of switch
            mu_home = pm.math.switch(mu_home > min_mu, mu_home, min_mu)
            mu_away = pm.math.switch(mu_away > min_mu, mu_away, min_mu)

            # observed
            pm.Poisson("obs_home_goals", observed=home_goals, mu=mu_home, dims="match")
            pm.Poisson("obs_away_goals", observed=away_goals, mu=mu_away, dims="match")

            # # with model1_home_advantage:
            football_idata = pm.sample()
            # posterior = trace.posterior.stack(sample=["chain", "draw"])
            # offence = posterior["offence"]
            # defence = posterior["defence"]
            # # home_advantage = posterior["home_advantage"]
            # # pm.predictions_to_inference_data

            # samples = pm.sample_posterior_predictive(trace)

    def _data_setter(
        self,
        X: Union[pd.DataFrame, pd.Series],
        goals: Union[pd.Series, np.ndarray],
    ) -> None:
        if isinstance(X, pd.DataFrame):
            x_values = X[["home_id", "away_id"]].values
        else:
            # Assuming "input" is the first column
            x_values = X[:, 0]

        with self.model:
            pm.set_data({"x_data": x_values})
            if goals is not None:
                pm.set_data(
                    {"goals": goals.values if isinstance(goals, pd.Series) else goals}
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
            "shape_priors": 18,  # toDo: check this for the number of teams ( kann wegen coords wohl raus )
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
            "draws": 2_000,
            "tune": 2_00,  # 10% of draws
            "chains": 1,
            "target_accept": 0.85,
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
        goals: Union[pd.Series, np.ndarray],
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """

        def _get_teams(df: pd.DataFrame) -> np.array:
            teams, uniques = pd.factorize(
                X[["home_id", "away_id"]].values.flatten(), sort=True
            )

            return teams, uniques

        self.model_coords = {
            "team": _get_teams(X)[1],
            "match": np.arange(len(X)),
            "field": ["home", "away"],
        }
        self.model_config = self.get_default_model_config()
        self.X = X
        goals_list = goals.tolist()
        # print("goals-shape: ", goals)

        self.y = pd.DataFrame(goals_list, columns=["home_goals", "away_goals"])
