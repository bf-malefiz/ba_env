"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.10
"""

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from kedro.pipeline import Pipeline, node, pipeline
from utils import create_pipelines_from_config, load_datasets_from_config

from .nodes_etl.nodes import (
    build_team_lexicon,
    extract_features,
    extract_toto,
    extract_x_data,
    extract_y_data,
    get_goal_results,
    vectorize_data,
)

# def load_datasets_from_config():
#     project_path = Path(__file__).parent.parent.parent.parent.parent
#     conf_path = str(project_path / settings.CONF_SOURCE)
#     conf_loader = OmegaConfigLoader(conf_source=conf_path)
#     return conf_loader["parameters"]["datasets"]


# def create_pipelines_from_config(dataset_list, base_pipeline):
#     pipelines = []
#     for dataset in dataset_list:
#         pipelines.append(
#             pipeline(
#                 base_pipeline,
#                 parameters={"params:model_parameters": "params:etl_pipeline"},
#                 namespace=dataset,
#             )
#         )

#     return pipeline(pipelines)


def create_pipeline(**kwargs) -> Pipeline:
    datasets = load_datasets_from_config()

    pipeline_instance = pipeline(
        pipe=[
            node(
                func=build_team_lexicon,
                inputs="CSV",
                outputs="team_lexicon",
                name="build_team_lexicon_node",
                tags=["etl_lexicon"],
            ),
            node(
                func=get_goal_results,
                inputs=["CSV", "team_lexicon"],
                outputs="goals",
                name="get_goal_results_node",
                tags=["etl_goals"],
                namespace=None,
            ),
            node(
                func=vectorize_data,
                inputs="goals",
                outputs="vectorized_data",
                name="vectorize_data_node",
                tags=["etl_vectorize_data"],
            ),
            node(
                func=extract_features,
                inputs=["vectorized_data", "params:model_parameters"],
                outputs="feature_data",
                name="extract_features_node",
                tags=["etl_features"],
            ),
            node(
                func=extract_x_data,
                inputs="feature_data",
                outputs="x_data",
                name="extract_x_data_node",
                tags=["etl_x_data"],
            ),
            node(
                func=extract_y_data,
                inputs=["vectorized_data", "params:model_parameters"],
                outputs="y_data",
                name="extract_y_data_node",
                tags=["etl_y_data"],
            ),
            node(
                func=extract_toto,
                inputs=["vectorized_data"],
                outputs="toto",
                name="extract_toto_node",
                tags=["etl_toto"],
            ),
        ],
    )

    return create_pipelines_from_config(
        dataset_list=datasets, pipeline_root="etl", base_pipeline=pipeline_instance
    )
