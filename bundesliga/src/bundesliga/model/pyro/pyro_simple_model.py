# import gin
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from bundesliga.model.pyro.pyro_model import PyroModel
from torch.distributions import constraints


# @gin.register
class SimplePyroModel(PyroModel):
    def __init__(self, team_lexicon, parameters, prior_diff=np.log(1.5)):
        super().__init__(team_lexicon, parameters)
        self.prior_diff = prior_diff

    def get_diffs(team_1, team_2, num_games=10000):
        mu_offence_param = pyro.param("mu_offence").data
        mu_defence_param = pyro.param("mu_defence").data
        sigma_offence_param = pyro.param("sigma_offence").data
        sigma_defence_param = pyro.param("sigma_defence").data

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

    def get_probs_winner_from_goal_results(self, team1, team2):
        diff_ij, diff_ji = SimplePyroModel.get_diffs(team1, team2)
        goals_of_team_1 = []
        goals_of_team_2 = []
        for r in diff_ij:
            goals_of_team_1.append(np.random.poisson(r))
        for r in diff_ji:
            goals_of_team_2.append(np.random.poisson(r))
        goals_of_team_1 = np.array(goals_of_team_1)
        goals_of_team_2 = np.array(goals_of_team_2)

        team1_wins = goals_of_team_1 > goals_of_team_2
        team2_wins = goals_of_team_1 < goals_of_team_2
        tie = goals_of_team_1 == goals_of_team_2

        p1 = team1_wins.sum() / len(team1_wins)
        p2 = team2_wins.sum() / len(team2_wins)
        tie = tie.sum() / len(tie)
        np.testing.assert_almost_equal(1, p1 + tie + p2)
        return np.array([tie, p1, p2])

    def predict_toto_probabilities(self, test_data, **kwargs):
        home_teams = test_data["home_id"].values
        away_teams = test_data["away_id"].values
        predicted_probabilities = list()
        for homeId, awayId in zip(home_teams, away_teams):
            predicted_probabilities.append(
                self.get_probs_winner_from_goal_results(homeId, awayId)
            )
        return np.array(predicted_probabilities)

    def get_model(self):
        nb_teams = len(self.team_lexicon)

        def _model(home_id, away_id, home_goals, away_goals, toto):
            nb_games = len(home_id)
            mu_offence_prior = torch.zeros(nb_teams) + self.prior_diff
            mu_defence_prior = torch.zeros(nb_teams)
            sigma_offence_prior = torch.ones(nb_teams)
            sigma_defence_prior = torch.ones(nb_teams)

            with pyro.plate("od_plate", size=nb_teams):
                offence = pyro.sample(
                    "offence", dist.Normal(mu_offence_prior, sigma_offence_prior)
                )
                defence = pyro.sample(
                    "defence", dist.Normal(mu_defence_prior, sigma_defence_prior)
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
        nb_teams = len(self.team_lexicon)

        def _guide(home_id, away_id, home_goals_, away_goals_, toto):
            # register the  variational parameters with Pyro.
            mu_offence = pyro.param(
                "mu_offence", torch.zeros(nb_teams) + self.prior_diff
            )
            mu_defence = pyro.param("mu_defence", torch.zeros(nb_teams))
            sigma_offence = pyro.param("sigma_offence", torch.ones(nb_teams) * 2.0)
            sigma_defence = pyro.param("sigma_defence", torch.ones(nb_teams) * 2.0)

            with pyro.plate("od_plate", size=nb_teams):
                offence = pyro.sample("offence", dist.Normal(mu_offence, sigma_offence))
                defence = pyro.sample("defence", dist.Normal(mu_defence, sigma_defence))

            return offence, defence

        return _guide
