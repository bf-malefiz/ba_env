"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_model_input_data, preprocess_league_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_league_data,
                inputs="D1_24-25",
                outputs=["preprocessed_league_data", "team_indices"],
                name="preprocess_league_data_node",
            ),
            node(
                func=create_model_input_data,
                inputs="preprocessed_league_data",
                outputs="model_input_table",
                name="model_input_data_node",
            ),
            # node(
            #     func=_get_teams,
            #     input="D1_24-25",
            #     outputs="preprocessed_teams",
            #     name="preprocessed_teams_node"
            # ),
            # node(
            #     func=_build_team_lexicon,
            #     input="D1_24-25",
            #     outputs="preprocessed_lexicon",
            #     name="preprocessed_lexicon_node"
            # ),
            # node(
            #     func=_get_goal_results,
            #     input="D1_24-25",
            #     outputs="preprocessed_goal_results",
            #     name="preprocessed_goal_results_node"
            # ),
            # node(
            #     func=_vectorized_data,
            #     input="D1_24-25",
            #     outputs="preprocessed_data_vectorized",
            #     name="preprocessed_data_vectorized_node"
            # ),
        ]
    )
