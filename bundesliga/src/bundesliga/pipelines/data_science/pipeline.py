"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline
from utils import load_config, split_time_data

from bundesliga import settings

# from .nodes_ml.posteriors import posterior_f1, posterior_f2, team_means
from .nodes_ml.train_model import fit, init_model, predict

# from .nodes_monitoring.plots import plot_goal_diffs, plot_offence_defence


def create_pipeline(**kwargs) -> Pipeline:
    datasets_list = load_config()["parameters"]["datasets"]

    pipes = []
    for namespace, variants in settings.DYNAMIC_PIPELINES_MAPPING.items():
        for variant in variants:
            for dataset_name in datasets_list:
                pipes.append(
                    pipeline(
                        pipeline(
                            create_walk_forward_pipeline(51),
                            namespace=dataset_name,
                        ),
                        parameters={
                            f"params:{dataset_name}.model_options": f"params:{namespace}.{variant}.model_options"
                        },
                        namespace=f"{namespace}",
                        tags=[variant, namespace],
                    )
                )

    return sum(pipes)


def create_pipelines_with_dataset_namespace(dataset_name, base_pipeline):
    pipelines = []
    pipelines.append(
        pipeline(
            base_pipeline,
            namespace=dataset_name,
            tags=dataset_name,
        )
    )

    return pipeline(pipelines)


def create_subpipeline_for_day(day: int) -> Pipeline:
    """
    Erzeugt Knoten [split -> fit -> predict -> evaluate] für Tag = `day`.
    """
    return Pipeline(
        [
            node(
                func=init_model,
                inputs={
                    "parameters": "params:model_options",
                },
                outputs=f"init_model_{day}",
                name=f"init_model_node_{day}",
            ),
            node(
                func=lambda: str(day),
                inputs=None,
                outputs=f"day_{day}",
                name=f"day_node_{day}",
            ),
            node(
                func=split_time_data,
                inputs={
                    "x_data": "x_data",
                    "y_data": "y_data",
                    "current_day": f"day_{day}",
                },
                outputs=[
                    f"x_data_fit_{day}",
                    f"y_data_fit_{day}",
                    f"x_data_pred_{day}",
                    f"y_data_pred_{day}",
                ],
                name=f"split_node_{day}",
            ),
            node(
                func=fit,
                inputs={
                    "x_data": f"x_data_fit_{day}",
                    "y_data": f"y_data_fit_{day}",
                    "model": f"init_model_{day}",
                    "team_lexicon": "team_lexicon",
                    "parameters": "params:model_options",
                    # "toto": f"toto_{day}",
                },
                outputs=[f"model_{day}", f"idata_{day}"],
                name=f"fit_node_{day}",
            ),
            node(
                func=predict,
                inputs={
                    "model": f"model_{day}",
                    "x_data": f"x_data_pred_{day}",
                    "idata": f"idata_{day}",
                    "parameters": "params:model_options",
                },
                outputs=f"results",
                name=f"predict_node_{day}",
            ),
        ],
    )


def create_walk_forward_pipeline(max_day: int) -> Pipeline:
    """
    Baut eine Pipeline, die für day=1..max_day Sub-Pipelines kaskadiert.
    => Ein einziger Pipeline-Lauf deckt alle Tage ab.
    """
    subpipelines = []
    for day in range(50, max_day + 1):
        if day % 3 == 0:
            subpipelines.append(create_subpipeline_for_day(day))

    # Sub-Pipelines zu einer großen Pipeline zusammenfassen
    # walk_forward_pipeline_collection = reduce(lambda p1, p2: p1 + p2, subpipelines)
    return subpipelines

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
    #         node(
    #             func=team_means,
    #             inputs="offence",
    #             outputs="offence_means",
    #             name="team_means_off_node",
    #             tags=["reporting", "training"],
    #         ),
    #         node(
    #             func=team_means,
    #             inputs="defence",
    #             outputs="defence_means",
    #             name="team_means_def_node",
    #             tags=["reporting", "training"],
    #         ),
    #         node(
    #             func=plot_offence_defence,
    #             inputs=["offence", "defence", "team_lexicon"],
    #             outputs="offence_defence_plot",
    #             name="plot_offence_defence_node",
    #             tags=["reporting", "training"],
    #         ),
    #         node(
    #             func=plot_goal_diffs,
    #             inputs=[
    #                 "offence",
    #                 "defence",
    #                 "team_lexicon",
    #                 "params:model_parameters",
    #             ],
    #             outputs="goal_diffs_plot",
    #             name="plot_goal_diffs_node",
    #             tags=["reporting", "training"],
    #         ),
    #     ]
    # )

    # active_pipe_f2 = pipeline(
    #     [pipe_fit, pipe_data_science_f2],
    #     inputs=["x_data", "y_data", "toto", "team_lexicon"],
    #     namespace="active_pipe_f2",
    #     outputs=[
    #         "fit_idata",
    #         "weights",
    #         "offence_defence_diff",
    #         "score",
    #         "home_advantage",
    #     ],
    #     parameters={"params:model_parameters": "params:f2_active_model_parameters"},
    #     # tags=["training"],
    # )
