from pathlib import Path

import numpy as np
import scipy.stats
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from kedro.pipeline import pipeline

min_mu = 0.0001
low = 10e-8  # Constant


def load_datasets_from_config():
    project_path = Path(__file__).parent.parent
    conf_path = str(project_path / settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path)
    return conf_loader["parameters"]["datasets"]


def create_pipelines_from_config(dataset_list, pipeline_root, base_pipeline):
    if pipeline_root == "etl":
        params = "params:etl_pipeline"
    elif pipeline_root == "training":
        params = "params:f1_active_model_parameters"

    pipelines = []
    for dataset in dataset_list:
        pipelines.append(
            pipeline(
                base_pipeline,
                parameters={"params:model_parameters": params},
                namespace=dataset,
            )
        )

    return pipeline(pipelines)


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
