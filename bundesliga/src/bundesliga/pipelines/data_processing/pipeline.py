"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline
from utils import load_config

from bundesliga import settings

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
    datasets_list = load_config()["parameters"]["datasets"]
    dataset_pipelines = []
    for dataset_name in datasets_list:
        dataset_pipelines.append(
            create_pipelines_with_dataset_namespace(dataset_name, data_processing)
        )

    pipes = []
    for namespace, _ in settings.DYNAMIC_PIPELINES_MAPPING.items():
        pipes.append(
            pipeline(
                dataset_pipelines,
                namespace=namespace,
                tags=settings.DYNAMIC_PIPELINES_MAPPING[namespace],
            )
        )
    return sum(pipes)


def create_pipelines_with_dataset_namespace(dataset_name, base_pipeline):
    pipelines = []
    pipelines.append(pipeline(base_pipeline, namespace=dataset_name, tags=dataset_name))

    return pipeline(pipelines)
