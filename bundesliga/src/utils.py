import numpy as np
import scipy.stats

min_mu = 0.0001
low = 10e-8  # Constant


def get_teams(model):
    return model.team_lexicon


def get_diffs(
    team_1, team_2, offence, defence
):  # toDO: offenc defence add in function from model1
    diff_ij = offence[team_1] - defence[team_2]
    diff_ji = offence[team_2] - defence[team_1]
    return diff_ij, diff_ji


def get_probs_winner(teamIndex1, teamIndex2):
    diff_ij, diff_ji = get_diffs(teamIndex1, teamIndex2)
    diff_ij[diff_ij < min_mu] = min_mu
    diff_ji[diff_ji < min_mu] = min_mu

    goals_of_team_1 = np.array([np.random.poisson(r) for r in diff_ij])
    goals_of_team_2 = np.array([np.random.poisson(r) for r in diff_ji])

    team1_wins = goals_of_team_1 > goals_of_team_2
    team2_wins = goals_of_team_1 < goals_of_team_2
    tie = goals_of_team_1 == goals_of_team_2

    p1 = team1_wins.mean()
    p2 = team2_wins.mean()
    tie = tie.mean()
    np.testing.assert_almost_equal(1.0, p1 + tie + p2)
    return tie, p1, p2


def get_goal_distribution(diff, max_goals=20):
    poisson_goals = np.zeros(max_goals)
    k = np.arange(0, max_goals)
    for lambda_ in diff:
        lambda_ = max(low, lambda_)
        poisson_goals += scipy.stats.poisson.pmf(k, lambda_)
    poisson_goals = poisson_goals / poisson_goals.sum()
    return poisson_goals
