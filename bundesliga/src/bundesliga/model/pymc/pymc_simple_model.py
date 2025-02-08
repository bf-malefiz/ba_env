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

import numpy as np
import pandas as pd
import pymc as pm

from bundesliga import settings
from bundesliga.model.pymc.pymc_model import PymcModel

min_mu = 0.0001
average_goals = 3.0

rng = np.random.default_rng(settings.SEED)


class SimplePymcModel(PymcModel):
    """
    A simple Bayesian model for forecasting football match outcomes using PyMC.

    This class builds a probabilistic model that estimates team-specific offensive and defensive strengths based on
    match data. By leveraging Poisson likelihoods, it models the number of goals scored in a match, ensuring that
    the computed mean values remain above a minimal threshold. The model is configurable through external model options,
    and it utilizes a team lexicon to map team identifiers to their respective attributes.

    Attributes:
        team_lexicon: A mapping of team identifiers to team names or indices used within the model.
        _model_type (str): A string identifier for the type of model implemented ("Simple_FootballModel").
        version (str): The version identifier of the SimplePymcModel implementation.
    """

    def __init__(self, model_options: dict, team_lexicon: pd.DataFrame) -> None:
        """
        Initialize the SimplePymcModel with model options and team lexicon.

        This constructor sets up the model by assigning a team lexicon, which is essential for mapping team identifiers
        to their corresponding indices or names. It then delegates to the base PymcModel's constructor for further
        initialization using the provided model options. This setup ensures that the model is properly configured to
        process input data and build the Bayesian model accordingly.

        Args:
            model_options (dict): A dictionary or configuration object containing parameters and settings specific to the model.
            team_lexicon (pd.DataFrame): A mapping that associates team identifiers with their respective names or indices.
        """
        self.team_lexicon = team_lexicon
        super().__init__(model_options, team_lexicon)

    _model_type = "Simple_FootballModel"
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
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
