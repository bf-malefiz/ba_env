"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import fit, plot_goal_diffs, plot_offence_defence, posterior, team_means


def create_pipeline(**kwargs) -> Pipeline:
    base_data_science = pipeline(
        [
            node(
                func=fit,
                inputs=["x_data", "y_data", "team_lexicon", "params:model_parameters"],
                outputs="",
                # outputs="fit_idata",
                name="fit_node",
            ),
            node(
                func=posterior,
                inputs="fit_idata",
                outputs=["offence", "defence"],
                name="posterior_node",
            ),
            node(
                func=team_means,
                inputs="offence",
                outputs="offence_means",
                name="team_means_off_node",
            ),
            node(
                func=team_means,
                inputs="defence",
                outputs="defence_means",
                name="team_means_def_node",
            ),
            node(
                func=plot_offence_defence,
                inputs=["offence", "defence", "team_lexicon"],
                outputs="offence_defence_plot",
                name="plot_offence_defence_node",
            ),
            node(
                func=plot_goal_diffs,
                inputs=[
                    "offence",
                    "defence",
                    "team_lexicon",
                    "params:model_parameters",
                ],
                outputs="goal_diffs_plot",
                name="plot_goal_diffs_node",
            ),
        ],
        # tags="tag1",  # Optional, each pipeline node will be tagged
        # namespace="",  # Optional
        # inputs={},  # Optional
        # outputs={},  # Optional
        # parameters={"params:model_parameters"},  # Optional
    )

    ds_pipeline_1 = pipeline(
        [
            base_data_science,
        ],
        inputs=["x_data", "y_data", "team_lexicon"],
        namespace="active_modelling_pipeline",
        outputs=["output_plot", "goal_diffs_plot"],
        parameters={"params:model_parameters": "params:active_model_parameters"},
    )

    # ds_pipeline_2 = pipeline(
    #     pipe=pipeline_instance,
    #     inputs="model_input_table",
    #     namespace="candidate_modelling_pipeline",
    # )

    # return ds_pipeline_1 + ds_pipeline_2
    return ds_pipeline_1
