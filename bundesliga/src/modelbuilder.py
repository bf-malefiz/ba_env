from typing import Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from model_interfaces import pymc_FootballModel
from numpy.random import RandomState
from pymc_experimental.model_builder import ModelBuilder

RANDOM_SEED = 8927
min_mu = 0.0001
average_goals = 3.0

rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")


class FootballModel(pymc_FootballModel):
    def __init__(self, model_config=None, sampler_config=None, team_lexicon=None):
        self.team_lexicon = team_lexicon
        super().__init__(model_config, sampler_config)

    # Give the model a name
    _model_type = "FootballModel_1"

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
            x_data_home = pm.Data(
                "x_data_home",
                X["home_id"].values,
                dims="match",
            )
            x_data_away = pm.Data(
                "x_data_away",
                X["away_id"].values,
                dims="match",
            )
            y_data_home = pm.Data(
                "y_data_home",
                goals["home_goals"].values,
                dims="match",
            )
            y_data_away = pm.Data(
                "y_data_away",
                goals["away_goals"].values,
                dims="match",
            )

            # prior parameters
            prior_params = self.model_config["prior_params"]
            off_mu = prior_params["off_mu"]
            def_mu = prior_params["def_mu"]
            off_tau = prior_params["off_tau"]
            def_tau = prior_params["def_tau"]

            # priors
            offence = pm.Normal(
                "offence",
                mu=off_mu,
                sigma=off_tau,
                # shape=shape_priors, ### kann ich auf shapes verzichten, wenn dims verwendet werden?
                dims="team",
            )
            defence = pm.Normal(
                "defence",
                mu=def_mu,
                tau=def_tau,
                dims="team",
            )

            # offence_home_away = offence[team_idx]
            # defence_home_away = defence[team_idx]
            # mu_home = mu_home_away[:, 0]
            # mu_away = mu_home_away[:, 1]
            # mu_home_away = offence_home_away - defence_home_away.eval()[:, [1, 0]]
            offence_home = offence[x_data_home]
            defence_home = defence[x_data_home]
            offence_away = offence[x_data_away]
            defence_away = defence[x_data_away]

            mu_home = offence_home - defence_away
            mu_away = offence_away - defence_home

            # # note: use exponent in practice instead of switch
            mu_home = pm.math.switch(mu_home > min_mu, mu_home, min_mu)
            mu_away = pm.math.switch(mu_away > min_mu, mu_away, min_mu)

            # observed
            pm.Poisson("home_goals", observed=y_data_home, mu=mu_home, dims="match")
            pm.Poisson("away_goals", observed=y_data_away, mu=mu_away, dims="match")

            # toDo: koennen wir die Poisson zusammenfuegen? goals ueber home+away
            # pm.Poisson(
            #     "home_points",
            #     observed=goals,
            #     mu=pt.stack((mu_home, mu_away)).T,
            #     dims=("match", "field"),
            # )


class FootballModel_2(pymc_FootballModel):
    def __init__(
        self, model_config=None, sampler_config=None, team_lexicon=None, toto=None
    ):
        self.team_lexicon = team_lexicon
        self.toto = toto
        super().__init__(model_config, sampler_config)

    # Give the model a name
    _model_type = "FootballModel_2"

    # And a version
    version = "0.1"

    def build_model(self, X: pd.DataFrame, goals: pd.Series, *args):
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
            # //////////////////////////////////////////////

            x_data_home = pm.Data(
                "x_data_home",
                X["home_id"].values,
                dims="match",
            )
            x_data_away = pm.Data(
                "x_data_away",
                X["away_id"].values,
                dims="match",
            )
            y_data_home = pm.Data(
                "y_data_home",
                goals["home_goals"].values,
                dims="match",
            )
            y_data_away = pm.Data(
                "y_data_away",
                goals["away_goals"].values,
                dims="match",
            )
            toto_data = pm.Data(
                "toto_data",
                self.toto.values.flatten(),
                dims="match",
            )
            # prior parameters
            prior_params = self.model_config["prior_params"]
            prior_score_params = prior_params["score"]
            prior_offence_defence_diff_params = prior_params["offence_defence_diff"]
            prior_home_advantage_params = prior_params["home_advantage"]
            weights = prior_params["weights"]
            # priors
            score = pm.Normal(
                "score",
                tau=prior_score_params.get("tau"),
                mu=prior_score_params.get("mu"),
                dims="team",
            )
            offence_defence_diff = pm.Normal(
                "offence_defence_diff",
                tau=prior_offence_defence_diff_params.get("tau"),
                mu=prior_offence_defence_diff_params.get("mu"),
                dims="team",
            )
            home_advantage = pm.Normal(
                "home_advantage",
                tau=prior_home_advantage_params.get("tau"),
                mu=prior_home_advantage_params.get("mu"),
            )

            # softmax regression weights for winner predicton:
            w_mu = tuple(weights.get("mu", (0.0, 0.0, 0.0)))
            weights = pm.Normal(
                "weights",
                mu=w_mu,
                tau=weights.get("tau", 0.0),
                shape=weights.get("shape", 3),
            )

            score_home = score[x_data_home] + home_advantage

            offence_home = score_home + offence_defence_diff[x_data_home]
            defence_home = score[x_data_home] - offence_defence_diff[x_data_home]
            offence_away = score[x_data_away] + offence_defence_diff[x_data_away]
            defence_away = score[x_data_away] - offence_defence_diff[x_data_away]

            mu_home = pm.math.exp(offence_home - defence_away)
            mu_away = pm.math.exp(offence_away - defence_home)
            # toDo: ungenutzte variablen?
            # home_value = pm.math.switch(mu_home < min_mu, min_mu, mu_home)
            # away_value = pm.math.switch(mu_away < min_mu, min_mu, mu_away)

            pm.Poisson("home_goals", observed=y_data_home, mu=mu_home, dims="match")
            pm.Poisson("away_goals", observed=y_data_away, mu=mu_away, dims="match")

            home_away_score_diff = score_home - score[x_data_away]
            home_away_score_diff = home_away_score_diff.reshape((-1, 1)).repeat(
                3, axis=1
            )

            pred = pm.math.exp(home_away_score_diff * weights)
            pred = (pred.T / pm.math.sum(pred, axis=1)).T
            pm.Categorical("toto", p=pred, observed=toto_data)
