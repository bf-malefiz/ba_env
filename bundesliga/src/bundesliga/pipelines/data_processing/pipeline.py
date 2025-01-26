"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.10
"""

from bundesliga import settings
from kedro.pipeline import Pipeline, node, pipeline

from .nodes_etl.nodes import (
    build_team_lexicon,
    get_goal_results,
    vectorize_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    data_processing = pipeline(
        pipe=[
            node(
                func=build_team_lexicon,
                inputs="CSV",
                outputs="team_lexicon",
                name="build_team_lexicon_node",
            ),
            node(
                func=get_goal_results,
                inputs=["CSV", "team_lexicon"],
                outputs="goals",
                name="get_goal_results_node",
                namespace=None,
            ),
            node(
                func=vectorize_data,
                inputs="goals",
                outputs="vectorized_data",
                name="vectorize_data_node",
            ),
        ],
    )
    datasets_list = settings.DATASETS
    dataset_pipelines = []

    for dataset_name in datasets_list:
        dataset_pipelines.append(
            pipeline(data_processing, namespace=dataset_name, tags=dataset_name)
        )
    return sum(dataset_pipelines)
