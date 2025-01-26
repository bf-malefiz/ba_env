"""Project pipelines."""
from typing import Dict

from bundesliga import __version__ as PROJECT_VERSION
from bundesliga import settings
from bundesliga.pipelines.data_processing.pipeline import (
    create_pipeline as create_etl_pipeline,
)
from bundesliga.pipelines.data_science.pipeline import eval_pipeline, ml_pipeline
from kedro.pipeline import Pipeline, pipeline
from utils import load_config


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipeline."""

    parameters = load_config()["parameters"]
    start_day = parameters["model_options"]["start_day"]
    walk_forward = parameters["model_options"]["walk_forward"]

    etl_pipeline = create_etl_pipeline()
    default_pipeline = []
    pipeline_dict = {"etl": etl_pipeline}

    for engine, variants in settings.DYNAMIC_PIPELINES_MAPPING.items():
        engine_pipeline_collection = []
        for variant in variants:
            for dataset_name in settings.DATASETS:
                engine_pipeline_collection.append(
                    pipeline(
                        ml_pipeline(
                            start_day=start_day,
                            walk_forward=walk_forward,
                            engine=engine,
                            variant=variant,
                            dataset_name=dataset_name,
                        ),
                        inputs={
                            "team_lexicon": f"{dataset_name}.team_lexicon",
                            "vectorized_data": f"{dataset_name}.vectorized_data",
                        },
                        namespace=f"{engine}",
                        tags=[engine],
                    )
                    + eval_pipeline(
                        startday=start_day,
                        walks=walk_forward,
                        setting=[
                            (engine, [variant]),
                        ],
                        dataset_name=dataset_name,
                    )
                )

        pipeline_dict[engine] = pipeline(engine_pipeline_collection)
        default_pipeline.append(pipeline(engine_pipeline_collection))

    pipeline_dict["__default__"] = pipeline(default_pipeline)
    # reporting_pipeline = ml_pipeline.only_nodes_with_tags("reporting")

    return pipeline_dict
