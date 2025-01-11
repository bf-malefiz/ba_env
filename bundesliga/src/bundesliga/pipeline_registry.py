"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from bundesliga import __version__ as PROJECT_VERSION
from bundesliga.pipelines.data_processing.pipeline import (
    create_pipeline as create_etl_pipeline,
)
from bundesliga.pipelines.data_science.pipeline import (
    create_pipeline as create_ml_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    etl_pipeline = create_etl_pipeline()
    ml_pipeline = create_ml_pipeline()
    reporting_pipeline = ml_pipeline.only_nodes_with_tags("reporting")

    return {
        "etl": etl_pipeline,
        "training": ml_pipeline,
        "reporting_pipeline": reporting_pipeline,
        "__default__": [etl_pipeline, ml_pipeline],
    }
