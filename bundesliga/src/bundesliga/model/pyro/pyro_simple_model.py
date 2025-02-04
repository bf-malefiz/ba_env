import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from bundesliga import settings
from bundesliga.model.pyro.pyro_model import PyroModel
from bundesliga.model.pyro.pyro_parameters import FootballParameters


class SimplePyroModel(PyroModel):
    """
    A simple Pyro-based model for predicting football match outcomes.

    This model extends the PyroModel class and implements a basic probabilistic model
    for predicting goals and match results. It uses Normal distributions for team strengths
    and Poisson distributions for goal predictions.

    Attributes:
        prior_diff (float): A prior value for the difference in team offensive strengths.
    """

    def __init__(self, team_lexicon, parameters, prior_diff=np.log(1.5)):
        """
        Initializes the SimplePyroModel with team lexicon, parameters, and a prior difference.

        Args:
            team_lexicon (dict): A dictionary mapping team names to unique IDs.
            parameters (dict): Configuration parameters for the model.
            prior_diff (float): A prior value for the difference in team offensive strengths.
        """
        super().__init__(team_lexicon, parameters)
        self.prior_diff = prior_diff
        self.parameter_module = FootballParameters(
            nb_teams=len(self.team_lexicon), prior_diff=self.prior_diff
        )

    def get_diffs(self, team_1, team_2, num_games=10000):
        """
        Calculates the differences in offensive and defensive strengths between two teams.

        Args:
            team_1 (int): The ID of the first team.
            team_2 (int): The ID of the second team.
            num_games (int): The number of simulations to run (default: 10000).

        Returns:
            tuple: Two arrays containing the differences in offensive and defensive strengths.
        """
        mu_offence_param = self.parameter_module.mu_offence.data
        mu_defence_param = self.parameter_module.mu_defence.data
        sigma_offence_param = self.parameter_module.sigma_offence.data
        sigma_defence_param = self.parameter_module.sigma_defence.data

        team1_offence_samples = np.random.normal(
            mu_offence_param[team_1], sigma_offence_param[team_1], num_games
        )
        team1_defence_samples = np.random.normal(
            mu_defence_param[team_1], sigma_defence_param[team_1], num_games
        )
        team2_offence_samples = np.random.normal(
            mu_offence_param[team_2], sigma_offence_param[team_2], num_games
        )
        team2_defence_samples = np.random.normal(
            mu_defence_param[team_2], sigma_defence_param[team_2], num_games
        )

        diff_ij = np.exp(team1_offence_samples - team2_defence_samples)
        diff_ji = np.exp(team2_offence_samples - team1_defence_samples)
        return diff_ij, diff_ji

    def predict_goals(self, test_data, **kwargs):
        """
        Predicts the goals scored by home and away teams based on test data.

        Args:
            test_data (pd.DataFrame): A DataFrame containing home and away team IDs.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: A DataFrame containing predicted home and away goals.
        """
        pyro.set_rng_seed(settings.SEED)
        np.random.seed(settings.SEED)

        team1 = test_data["home_id"].values
        team2 = test_data["away_id"].values

        diff_ij, diff_ji = self.get_diffs(team1, team2)
        goals_of_team_1 = []
        goals_of_team_2 = []
        for r in diff_ij:
            goals_of_team_1.append(np.random.poisson(r))
        for r in diff_ji:
            goals_of_team_2.append(np.random.poisson(r))
        goals_of_team_1 = np.array(goals_of_team_1)
        goals_of_team_2 = np.array(goals_of_team_2)

        df = pd.DataFrame(
            {"home_goals": goals_of_team_1, "away_goals": goals_of_team_2}
        )

        # xarr = df.to_xarray()

        return df

    def get_model(self):
        """
        Defines the probabilistic model for team offensive and defensive strengths.

        Returns:
            function: A function representing the Pyro model.
        """
        nb_teams = len(self.team_lexicon)

        def _model(home_id, away_id, home_goals, away_goals, toto):
            nb_games = len(home_id)
            pyro.module("football_module", self.parameter_module)
            # mu_offence_prior = torch.zeros(nb_teams) + self.prior_diff
            # mu_defence_prior = torch.zeros(nb_teams)
            # sigma_offence_prior = torch.ones(nb_teams)
            # sigma_defence_prior = torch.ones(nb_teams)

            with pyro.plate("od_plate", size=nb_teams):
                offence = pyro.sample(
                    "offence",
                    dist.Normal(
                        self.parameter_module.mu_offence_prior,
                        self.parameter_module.sigma_offence_prior.clamp(min=1e-5),
                    ),
                )
                defence = pyro.sample(
                    "defence",
                    dist.Normal(
                        self.parameter_module.mu_defence_prior,
                        self.parameter_module.sigma_defence_prior.clamp(min=1e-5),
                    ),
                )

            home_values = torch.exp(offence[home_id] - defence[away_id])
            away_values = torch.exp(offence[home_id] - defence[away_id])

            with pyro.plate("observed_data", size=nb_games) as ind:
                pyro.sample(
                    "home_goals",
                    dist.Poisson(home_values.index_select(0, ind)),
                    obs=torch.tensor(home_goals),
                )
                pyro.sample(
                    "away_goals",
                    dist.Poisson(away_values.index_select(0, ind)),
                    obs=torch.tensor(away_goals),
                )

            return offence, defence

        return _model

    def get_guide(self):
        """
        Defines the guide (variational distribution) for variational inference.

        Returns:
            function: A function representing the Pyro guide.
        """
        nb_teams = len(self.team_lexicon)

        def _guide(home_id, away_id, home_goals_, away_goals_, toto):
            pyro.module("football_module", self.parameter_module)
            # register the  variational parameters with Pyro.
            # mu_offence = pyro.param(
            #     "mu_offence", torch.zeros(nb_teams) + self.prior_diff
            # )
            # mu_defence = pyro.param("mu_defence", torch.zeros(nb_teams))
            # sigma_offence = pyro.param("sigma_offence", torch.ones(nb_teams) * 2.0)
            # sigma_defence = pyro.param("sigma_defence", torch.ones(nb_teams) * 2.0)

            with pyro.plate("od_plate", size=nb_teams):
                offence = pyro.sample(
                    "offence",
                    dist.Normal(
                        self.parameter_module.mu_offence,
                        self.parameter_module.sigma_offence.clamp(min=1e-5),
                    ),
                )
                defence = pyro.sample(
                    "defence",
                    dist.Normal(
                        self.parameter_module.mu_defence,
                        self.parameter_module.sigma_defence.clamp(min=1e-5),
                    ),
                )

            return offence, defence

        return _guide
