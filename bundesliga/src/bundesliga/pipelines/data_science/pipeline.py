"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from kedro.pipeline import Pipeline, node, pipeline
from utils import create_pipelines_from_config, load_datasets_from_config

from .nodes_ml.posteriors import posterior_f1, posterior_f2, team_means
from .nodes_ml.train_model import fit, predict
from .nodes_monitoring.plots import plot_goal_diffs, plot_offence_defence

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
#                 parameters={
#                     "params:model_parameters": "params:f1_active_model_parameters"
#                 },
#                 namespace=dataset,
#             )
#         )

#     return pipeline(pipelines)


def create_pipeline(**kwargs) -> Pipeline:
    datasets = load_datasets_from_config()

    pipeline_training = pipeline(
        [
            node(
                func=fit,
                inputs={
                    "x_data": "x_data",
                    "y_data": "y_data",
                    "team_lexicon": "team_lexicon",
                    "parameters": "params:model_parameters",
                    "toto": "toto",
                },
                outputs="model",
                name="fit_node",
                tags=["training"],
            ),
        ]
    )
    pipeline_predict = pipeline(
        [
            node(
                func=predict,
                inputs={
                    "model": "model",
                    "x_data": "x_data",
                    "parameters": "params:model_parameters",
                },
                outputs="idata",
                name="predict_node",
                tags=["training"],
            ),
        ]
    )

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

    active_pipe_f1 = pipeline_training + pipeline_predict

    return create_pipelines_from_config(
        dataset_list=datasets, pipeline_root="training", base_pipeline=active_pipe_f1
    )
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
