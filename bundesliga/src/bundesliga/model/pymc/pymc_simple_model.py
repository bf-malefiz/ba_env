import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from bundesliga import settings
from bundesliga.model.pymc.pymc_model import PymcModel

min_mu = 0.0001
average_goals = 3.0

rng = np.random.default_rng(settings.SEED)
az.style.use("arviz-darkgrid")


class SimplePymcModel(PymcModel):
    def __init__(self, model_options, team_lexicon):
        self.team_lexicon = team_lexicon
        super().__init__(model_options, team_lexicon)

    # Give the model a name
    _model_type = "Simple_FootballModel"

    # And a version
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
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
        if isinstance(X, pd.DataFrame):
            X = X[["home_id", "away_id"]]

        y = y.values if isinstance(y, pd.Series) else y
        self._generate_and_preprocess_model_data(X, y)

        with pm.Model(coords=self.model_coords) as self.model:
            # Create mutable data containers
            x_data_home = pm.Data(
                "x_data_home",
                self.X["home_id"].values,
                dims="match",
            )
            x_data_away = pm.Data(
                "x_data_away",
                self.X["away_id"].values,
                dims="match",
            )

            y_data = pm.Data(
                "y_data",
                self.y,
                dims=["match", "y"],
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
                dims="team",
            )
            defence = pm.Normal(
                "defence",
                mu=def_mu,
                tau=def_tau,
                dims="team",
            )

            offence_home = offence[x_data_home]
            defence_home = defence[x_data_home]
            offence_away = offence[x_data_away]
            defence_away = defence[x_data_away]

            mu_home = offence_home - defence_away
            mu_away = offence_away - defence_home

            # # note: use exponent in practice instead of switch
            mu_home = pm.math.switch(mu_home > min_mu, mu_home, min_mu)
            mu_away = pm.math.switch(mu_away > min_mu, mu_away, min_mu)

            mu = pm.math.stack([mu_home, mu_away], axis=-1)

            # observed
            # pm.Poisson("home_goals", observed=y_data_home, mu=mu_home, dims="match")
            # pm.Poisson("away_goals", observed=y_data_away, mu=mu_away, dims="match")
            pm.Poisson("goals", observed=y_data, mu=mu, dims=["match", "y"])
