"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_team_lexicon,
    extract_features,
    extract_x_data,
    extract_y_data,
    get_goal_results,
    vectorize_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    dataset = "D1_24-25"
    pipeline_instance = pipeline(
        [
            node(
                func=build_team_lexicon,
                inputs=dataset,
                outputs="team_lexicon",
                name="build_team_lexicon_node",
            ),
            node(
                func=get_goal_results,
                inputs=[dataset, "team_lexicon"],
                outputs="goals",
                name="get_goal_results_node",
            ),
            node(
                func=vectorize_data,
                inputs="goals",
                outputs="vectorized_data",
                name="vectorize_data_node",
            ),
            node(
                func=extract_features,
                inputs=["vectorized_data", "params:model_parameters"],
                outputs="feature_data",
                name="extract_features_node",
            ),
            node(
                func=extract_x_data,
                inputs="feature_data",
                outputs="x_data",
                name="extract_x_data_node",
            ),
            node(
                func=extract_y_data,
                inputs=["vectorized_data", "params:model_parameters"],
                outputs="y_data",
                name="extract_y_data_node",
            ),
        ]
    )

    active_pp_pipeline = pipeline(
        pipe=pipeline_instance,
        inputs=dataset,
        outputs=[
            "x_data",
            "y_data",
            "feature_data",
            "vectorized_data",
            "goals",
            "team_lexicon",
        ],
        namespace="active_pp_pipeline",
    )
    return active_pp_pipeline
