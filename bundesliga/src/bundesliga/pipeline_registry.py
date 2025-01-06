"""Project pipelines."""
from platform import python_version
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kedro_mlflow.pipeline import pipeline_ml_factory

from bundesliga import __version__ as PROJECT_VERSION
from bundesliga.pipelines.data_processing.pipeline import (
    create_pipeline as dp_create_pipeline,
)
from bundesliga.pipelines.data_science.pipeline import (
    create_pipeline as ds_create_pipeline,
)

# def register_pipelines() -> dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines


def register_pipelines() -> Dict[str, Pipeline]:
    dp_pipeline = dp_create_pipeline()
    ds_pipeline = ds_create_pipeline()
    ml_pipeline = dp_pipeline + ds_pipeline
    inference_pipeline = ml_pipeline.only_nodes_with_tags("inference")
    training_pipeline_ml = pipeline_ml_factory(
        training=ml_pipeline.only_nodes_with_tags("training"),
        inference=inference_pipeline,
        input_name="x_data",
        log_model_kwargs=dict(
            artifact_path="kedro_mlflow_test",
            conda_env={
                "python": 3.12,
                "build_dependencies": ["pip"],
                "dependencies": [f"bundesliga=={PROJECT_VERSION}"],
            },
            signature="auto",
        ),
    )
    return {"training": training_pipeline_ml, "__default__": training_pipeline_ml}
