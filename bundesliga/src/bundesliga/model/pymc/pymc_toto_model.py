"""
Summary:
    Implementation class for PyMC-based football match prediction models.

Dependencies:
    - numpy: For numerical computations and random number generation.
    - pandas: For data manipulation and handling of input datasets.
    - pymc: For probabilistic programming and Bayesian model specification.
    - bundesliga.settings: For configuration parameters, including the random seed.
    - bundesliga.model.pymc.pymc_model: The base class (PymcModel) that this model extends.
"""

import pandas as pd
import pymc as pm

from bundesliga.model.pymc.pymc_model import PymcModel


class TotoModel(PymcModel):
    """
    A simple Bayesian model for forecasting football match outcomes using PyMC.

    This class builds a probabilistic model that estimates team-specific offensive and defensive strengths based on
    match data. By leveraging Poisson likelihoods, it models the number of goals scored in a match, ensuring that
    the computed mean values remain above a minimal threshold. The model is configurable through external model options,
    and it utilizes a team lexicon to map team identifiers to their respective attributes.

    Attributes:
        team_lexicon: A mapping of team identifiers to team names or indices used within the model.
        _model_type (str): A string identifier for the type of model implemented ("Simple_FootballModel").
        version (str): The version identifier of the Totomodel implementation.
    """

    def __init__(
        self,
        model_options,
        team_lexicon=None,
        toto=None,
    ):
        self.team_lexicon = team_lexicon
        self.toto = toto
        super().__init__(model_options, team_lexicon)

    # Give the model a name
    _model_type = "Toto_FootballModel"

    # And a version
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, *args):
        """
        Build and configure the PyMC probabilistic model for predicting football match outcomes.

        This method prepares and constructs the Bayesian model by first preprocessing the input data. It extracts the
        necessary features (specifically, the home and away team identifiers) from the provided DataFrame, and converts
        the target data into an appropriate format. The method then sets up mutable data containers within a PyMC model
        context to hold these inputs. Using prior parameters from the model configuration (such as offensive and defensive
        means and variances), it defines normal priors for team strengths. The model calculates expected goal rates for both
        home and away teams, ensuring that these rates do not fall below a minimum threshold by using a conditional switch.
        Finally, the observed match outcomes (goals) are modeled using a Poisson likelihood, thereby completing the model
        specification.

        Args:
            X (pd.DataFrame): The input DataFrame containing match features. It must include 'home_id' and 'away_id' columns
                              that denote the identifiers for the home and away teams, respectively.
            y (pd.Series): The target data representing observed goal counts, provided as a pandas Series.
            **kwargs: Additional keyword arguments for customizing the model configuration, if necessary.
        """

        if isinstance(X, pd.DataFrame):
            X = X[["home_id", "away_id"]]

        y = y.values if isinstance(y, pd.Series) else y
        self._generate_and_preprocess_model_data(X, y)

        with pm.Model(coords=self.model_coords) as self.model:
            # //////////////////////////////////////////////

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

            toto_data = pm.Data(
                "toto_data",
                self.toto,
                dims="match",
            )
            # prior parameters
            prior_params = self.model_config["prior_params"]
            prior_score_params = prior_params["score"]
            prior_offence_defence_diff_params = prior_params["offence_defence_diff"]
            prior_home_advantage_params = prior_params["home_advantage"]
            weights = prior_params["weights"]

            score_tau = prior_score_params["tau"]
            score_mu = prior_score_params["mu"]

            offence_defence_diff_tau = prior_offence_defence_diff_params["tau"]
            offence_defence_diff_mu = prior_offence_defence_diff_params["mu"]

            home_advantage_tau = prior_home_advantage_params["tau"]
            home_advantage_mu = prior_home_advantage_params["mu"]

            # softmax regression weights for winner predicton:
            w_mu = tuple(weights.get("mu", (0.0, 0.0, 0.0)))

            # priors
            score = pm.Normal(
                "score",
                tau=score_tau,
                mu=score_mu,
                dims="team",
            )
            offence_defence_diff = pm.Normal(
                "offence_defence_diff",
                tau=offence_defence_diff_tau,
                mu=offence_defence_diff_mu,
                dims="team",
            )
            home_advantage = pm.Normal(
                "home_advantage",
                tau=home_advantage_tau,
                mu=home_advantage_mu,
            )

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
            mu = pm.math.stack([mu_home, mu_away], axis=-1)
            # toDo: ungenutzte variablen?
            # home_value = pm.math.switch(mu_home < min_mu, min_mu, mu_home)
            # away_value = pm.math.switch(mu_away < min_mu, min_mu, mu_away)

            # pm.Poisson("home_goals", observed=y_data_home, mu=mu_home, dims="match")
            # pm.Poisson("away_goals", observed=y_data_away, mu=mu_away, dims="match")
            pm.Poisson("goals", observed=y_data, mu=mu, dims=["match", "y"])

            home_away_score_diff = score_home - score[x_data_away]
            home_away_score_diff = home_away_score_diff.reshape((-1, 1)).repeat(
                3, axis=1
            )

            pred = pm.math.exp(home_away_score_diff * weights)
            pred = (pred.T / pm.math.sum(pred, axis=1)).T
            pm.Categorical("toto", p=pred, observed=toto_data, dims="match")
