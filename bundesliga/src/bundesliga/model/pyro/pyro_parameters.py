import torch
from torch import nn


class FootballParameters(nn.Module):
    def __init__(self, nb_teams, prior_diff):
        super().__init__()
        # Variational Parameter (z.B. Mittelwerte, Offence)
        self.mu_offence_prior = nn.Parameter(torch.zeros(nb_teams) + prior_diff)
        self.mu_defence_prior = nn.Parameter(torch.zeros(nb_teams))

        # Variational Parameter (z.B. Standardabweichungen)
        self.sigma_offence_prior = nn.Parameter(torch.ones(nb_teams))
        self.sigma_defence_prior = nn.Parameter(torch.ones(nb_teams))

        self.mu_offence = nn.Parameter(torch.zeros(nb_teams) + prior_diff)
        self.mu_defence = nn.Parameter(torch.zeros(nb_teams))
        self.sigma_offence = nn.Parameter(torch.ones(nb_teams) * 2.0)
        self.sigma_defence = nn.Parameter(torch.ones(nb_teams) * 2.0)
