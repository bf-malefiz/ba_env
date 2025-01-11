"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.10
"""

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from kedro.pipeline import Pipeline, node, pipeline

from .nodes_etl.nodes import (
    build_team_lexicon,
    extract_features,
    extract_toto,
    extract_x_data,
    extract_y_data,
    get_goal_results,
    vectorize_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    project_path = Path(__file__).parent.parent.parent.parent.parent
    conf_path = str(project_path / settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path)
    datasets = conf_loader["parameters"]["datasets"]

    # dataset = "D1_24-25"

    pipeline_collection = None

    for dataset in datasets:
        pipeline_instance = pipeline(
            pipe=[
                node(
                    func=build_team_lexicon,
                    inputs=str(dataset + "#CSV"),
                    outputs=f"{dataset}.team_lexicon",
                    name=f"{dataset}.build_team_lexicon_node",
                    tags=["etl_lexicon"],
                ),
                node(
                    func=get_goal_results,
                    inputs=[str(dataset + "#CSV"), f"{dataset}.team_lexicon"],
                    outputs=f"{dataset}.goals",
                    name=f"{dataset}.get_goal_results_node",
                    tags=["etl_goals"],
                    namespace=None,
                ),
                node(
                    func=vectorize_data,
                    inputs=f"{dataset}.goals",
                    outputs=f"{dataset}.vectorized_data",
                    name=f"{dataset}.vectorize_data_node",
                    tags=["etl_vectorize_data"],
                ),
                node(
                    func=extract_features,
                    inputs=[f"{dataset}.vectorized_data", "params:model_parameters"],
                    outputs=f"{dataset}.feature_data",
                    name=f"{dataset}.extract_features_node",
                    tags=["etl_features"],
                ),
                node(
                    func=extract_x_data,
                    inputs=f"{dataset}.feature_data",
                    outputs=f"{dataset}.x_data",
                    name=f"{dataset}.extract_x_data_node",
                    tags=["etl_x_data"],
                ),
                node(
                    func=extract_y_data,
                    inputs=[f"{dataset}.vectorized_data", "params:model_parameters"],
                    outputs=f"{dataset}.y_data",
                    name=f"{dataset}.extract_y_data_node",
                    tags=["etl_y_data"],
                ),
                node(
                    func=extract_toto,
                    inputs=[f"{dataset}.vectorized_data"],
                    outputs=f"{dataset}.toto",
                    name=f"{dataset}.extract_toto_node",
                    tags=["etl_toto"],
                ),
            ],
            parameters={"params:model_parameters": "params:etl_pipeline"},
        )

    if pipeline_collection:
        pipeline_collection = pipeline_collection + pipeline_instance
    else:
        pipeline_collection = pipeline_instance

    return pipeline_collection
