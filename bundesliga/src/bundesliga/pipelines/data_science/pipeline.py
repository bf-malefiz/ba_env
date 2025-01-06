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
    posterior,
    team_means,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe_init_model = pipeline(
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
        ]
    )
    # init_model2 = pipeline(
    #     [
    #         node(
    #             func=init_model,
    #             inputs={
    #                 "team_lexicon": "team_lexicon",
    #                 "parameters": "params:model_parameters",
    #                 "toto": "toto",
    #             },
    #             outputs="tmp_model",
    #             name="init_model_node",
    #         ),
    #     ]
    # )
    base_data_science = pipeline(
        [
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
        ],
    )
    reporting = pipeline(
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

    # ds_pipeline_1 = pipeline(
    #     [init_model1, base_data_science, reporting],
    #     inputs=["x_data", "y_data", "team_lexicon"],
    #     namespace="active_modelling_pipeline",
    #     outputs=["offence_defence_plot", "goal_diffs_plot"],
    #     parameters={"params:model_parameters": "params:active_model_parameters"},
    #     tags="active_modelling",
    # )
    ds_pipeline_1 = pipeline(
        [pipe_init_model, base_data_science, reporting],
        inputs=["x_data", "y_data", "toto", "team_lexicon"],
        namespace="active_modelling_pipeline",
        outputs=["offence_defence_plot", "goal_diffs_plot"],
        parameters={"params:model_parameters": "params:active_model_parameters"},
        tags="active_modelling",
    )
    # ds_pipeline_2 = pipeline(
    #     pipe=pipeline_instance,
    #     inputs="model_input_table",
    #     namespace="candidate_modelling_pipeline",
    # )

    # return ds_pipeline_1 + ds_pipeline_2
    return ds_pipeline_1
