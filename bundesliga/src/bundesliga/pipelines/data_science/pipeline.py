"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    fit,
    init_model,
    plot_goal_diffs,
    plot_offence_defence,
    posterior_f1,
    posterior_f2,
    team_means,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe_fit = pipeline(
        [
            node(
                func=init_model,
                inputs={
                    "team_lexicon": "team_lexicon",
                    "parameters": "params:model_parameters",
                    "toto": "toto",
                },
                outputs="tmp_model",
                name="init_model_node",
            ),
            node(
                func=fit,
                inputs=[
                    "tmp_model",
                    "x_data",
                    "y_data",
                    "team_lexicon",
                    "params:model_parameters",
                ],
                # outputs="",
                outputs="fit_idata",
                name="fit_node",
            ),
        ]
    )

    pipe_data_science_f1 = pipeline(
        [
            node(
                func=posterior_f1,
                inputs="fit_idata",
                outputs=[
                    "offence",
                    "defence",
                ],
                name="posterior_f1_node",
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
        ],
    )

    pipe_data_science_f2 = pipeline(
        [
            node(
                func=posterior_f2,
                inputs="fit_idata",
                outputs=[
                    "weights",
                    "offence_defence_diff",
                    "score",
                    "home_advantage",
                ],
                name="posterior_f2_node",
            ),
            # node(
            #     func=team_means,
            #     inputs="offence",
            #     outputs="offence_means",
            #     name="team_means_off_node",
            # ),
            # node(
            #     func=team_means,
            #     inputs="defence",
            #     outputs="defence_means",
            #     name="team_means_def_node",
            # ),
        ],
    )

    reporting_f1 = pipeline(
        [
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
        ]
    )

    active_pipe_f1 = pipeline(
        [pipe_fit, pipe_data_science_f1, reporting_f1],
        inputs=["x_data", "y_data", "toto", "team_lexicon"],
        namespace="active_pipe_f1",
        outputs=["offence_defence_plot", "goal_diffs_plot"],
        parameters={"params:model_parameters": "params:f1_active_model_parameters"},
        tags="active_pipe_f1",
    )
    active_pipe_f2 = pipeline(
        [pipe_fit, pipe_data_science_f2],
        inputs=["x_data", "y_data", "toto", "team_lexicon"],
        namespace="active_pipe_f2",
        outputs=[
            "fit_idata",
            "weights",
            "offence_defence_diff",
            "score",
            "home_advantage",
        ],
        parameters={"params:model_parameters": "params:f2_active_model_parameters"},
        tags="active_pipe_f2",
    )

    return active_pipe_f1 + active_pipe_f2
