# """Project pipelines."""
# from platform import python_version
# from typing import Dict

# from kedro.framework.project import find_pipelines
# from kedro.pipeline import Pipeline
# from kedro_mlflow.pipeline import pipeline_ml_factory

# from bundesliga import __version__ as PROJECT_VERSION
# from bundesliga.pipelines.data_processing.pipeline import (
#     create_pipeline as create_etl_pipeline,
# )
# from bundesliga.pipelines.data_science.pipeline import (
#     create_pipeline as create_ml_pipeline,
# )

# # def register_pipelines() -> dict[str, Pipeline]:
# #     """Register the project's pipelines.

# #     Returns:
# #         A mapping from pipeline names to ``Pipeline`` objects.
# #     """
# #     pipelines = find_pipelines()
# #     pipelines["__default__"] = sum(pipelines.values())
# #     return pipelines


# def register_pipelines() -> Dict[str, Pipeline]:
#     etl_pipeline = create_etl_pipeline()
#     etl_lexicon_pipeline = etl_pipeline.only_nodes_with_tags("etl_lexicon")
#     etl_goals_pipeline = etl_pipeline.only_nodes_with_tags("etl_goals")
#     etl_vectorize_data_pipeline = etl_pipeline.only_nodes_with_tags(
#         "etl_vectorize_data"
#     )
#     etl_features_pipeline = etl_pipeline.only_nodes_with_tags("etl_features")
#     etl_x_data_pipeline = etl_pipeline.only_nodes_with_tags("etl_x_data")
#     etl_y_data_pipeline = etl_pipeline.only_nodes_with_tags("etl_y_data")
#     etl_toto_pipeline = etl_pipeline.only_nodes_with_tags("etl_toto")

#     ml_pipeline = create_ml_pipeline()
#     inference_pipeline = ml_pipeline.only_nodes_with_tags("inference")
#     training_pipeline_ml = pipeline_ml_factory(
#         training=ml_pipeline.only_nodes_with_tags("training"),
#         inference=inference_pipeline,
#         input_name="x_data",
#         log_model_kwargs=dict(
#             artifact_path="kedro_mlflow_test",
#             conda_env={
#                 "python": python_version(),
#                 "build_dependencies": ["pip"],
#                 "dependencies": [f"bundesliga=={PROJECT_VERSION}"],
#             },
#             signature="auto",
#         ),
#     )
#     reporting_pipeline = ml_pipeline.only_nodes_with_tags("reporting")
#     return {
#         "etl_lexicon_pipeline": etl_lexicon_pipeline,
#         "etl_goals_pipeline": etl_goals_pipeline,
#         "etl_vectorize_data_pipeline": etl_vectorize_data_pipeline,
#         "etl_features_pipeline": etl_features_pipeline,
#         "etl_x_data_pipeline": etl_x_data_pipeline,
#         "etl_y_data_pipeline": etl_y_data_pipeline,
#         "etl_toto_pipeline": etl_toto_pipeline,
#         "training": training_pipeline_ml,
#         "inference": inference_pipeline,
#         "reporting_pipeline": reporting_pipeline,
#         "__default__": etl_lexicon_pipeline
#         + etl_goals_pipeline
#         + etl_vectorize_data_pipeline
#         + etl_features_pipeline
#         + etl_x_data_pipeline
#         + etl_y_data_pipeline
#         + etl_toto_pipeline
#         + training_pipeline_ml
#         + inference_pipeline
#         + reporting_pipeline,
#     }
"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines