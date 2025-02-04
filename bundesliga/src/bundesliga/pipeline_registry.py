"""Project pipelines."""
from typing import Dict, List

from kedro.pipeline import Pipeline, pipeline
from kedro_datasets.pandas import CSVDataset

from bundesliga import __version__ as PROJECT_VERSION
from bundesliga import settings
from bundesliga.pipelines.data_processing.pipeline import (
    create_pipeline as create_etl_pipeline,
)
from bundesliga.pipelines.data_science.pipeline import eval_pipeline, ml_pipeline
from bundesliga.utils.utils import load_config


def read_matchlength_from_data(dataset_name: str, start_match: int) -> int:
    """
    Read the walk forward parameter from the data catalog.

    Returns:
        int: The walk forward parameter read from the data catalog.
    """
    data = CSVDataset(
    filepath=f"./data/01_raw/football-datasets/{dataset_name}.csv",
    load_args=dict(sep=",", encoding= "cp1252" ,index_col= 0)
)
    df = data.load()
    total_matches = len(df)
    if start_match >= total_matches:
        raise ValueError(
            f"Start match {start_match} is greater than or equal the number of matches in the dataset {len(df)}."
        )
    return total_matches - start_match -2

def build_engine_pipelines(
    engine: str, variants: List[str], start_match: int, last_match: int
) -> Pipeline:
    """
    Builds a collection of machine learning pipelines for a specific engine and its variants.

    This function creates a pipeline for each combination of engine, variant, and dataset.
    It combines the machine learning pipeline (`ml_pipeline`) with the evaluation pipeline (`eval_pipeline`)
    for each dataset specified in `settings.DATASETS`.

    Args:
        engine (str): The engine type (e.g., "pymc", "pyro").
        variants (List[str]): A list of variants for the engine.
        start_match (int): The starting match for the data.
        walk_forward (int): The number of matchs to walk forward in the data.

    Returns:
        Pipeline: A Kedro pipeline object representing the combined ML and evaluation pipelines
                 for the given engine and variants.

    Example:
        To build pipelines for the "pymc" engine with variants ["simple", "toto"]:
        ```python
        engine_pipeline = build_engine_pipelines(
            "pymc", ["simple", "toto"], start_match=20, walk_forward=5
        )
        ```
    """

    if not engine:
        raise ValueError("Engine type cannot be empty.")

    if not variants:
        raise ValueError("Variants list cannot be empty.")

    if not hasattr(settings, "DATASETS") or not settings.DATASETS:
        raise ValueError("Missing or empty 'DATASETS' in settings.")

    engine_pipelines = []
    is_last_match_set = last_match is not None
    for variant in variants:
        for dataset_name in settings.DATASETS:

            if not is_last_match_set:
                last_match = read_matchlength_from_data(dataset_name, start_match=start_match)
            if start_match is None:
                start_match = 0
            try:
                pipeline_obj = pipeline(
                    ml_pipeline(
                        start_match=start_match,
                        last_match=last_match,
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
                ) + eval_pipeline(
                    startmatch=start_match,
                    last_match=last_match,
                    setting=[(engine, [variant])],
                    dataset_name=dataset_name,
                )
                engine_pipelines.append(pipeline_obj)

            except Exception as e:
                raise ValueError(
                    f"Failed to build pipeline for engine '{engine}', variant '{variant}', "
                    f"and dataset '{dataset_name}': {str(e)}"
                )
            if not is_last_match_set:
                last_match = None
    return pipeline(engine_pipelines)


def register_pipelines() -> Dict[str, Pipeline]:
    """
    Registers the project's pipelines.

    This function dynamically creates and registers pipelines based on the configuration
    provided in `settings.DYNAMIC_PIPELINES_MAPPING`. It combines the ETL pipeline with
    machine learning and evaluation pipelines for each engine and variant.

    Returns:
        Dict[str, Pipeline]: A dictionary of pipeline names and their corresponding Pipeline objects.
                             The keys include:
                             - "etl": The ETL pipeline for data processing.
                             - "<engine>": The combined pipeline for a specific engine (e.g., "pymc", "pyro").
                             - "__default__": The default pipeline, which includes the ETL pipeline and all
                               machine learning pipelines.

    Raises:
        ValueError: If the required parameters or settings are missing or invalid.

    Example:
        To access the default pipeline:
        ```python
        from kedro.framework.project import pipelines

        pipeline = pipelines["__default__"]
        pipeline.run()
        ```
    """

    parameters = load_config()["parameters"]
    start_match = parameters["model_options"]["start_match"]
    last_match = parameters["model_options"]["last_match"]

    etl_pipeline = create_etl_pipeline()
    pipeline_dict = {"etl": etl_pipeline}

    default_pipelines = []

    if start_match is None:
        raise ValueError("Missing required parameters: 'start_match'.")
    if (
        not hasattr(settings, "DYNAMIC_PIPELINES_MAPPING")
        or not settings.DYNAMIC_PIPELINES_MAPPING
    ):
        raise ValueError("Missing or empty 'DYNAMIC_PIPELINES_MAPPING' in settings.")

    if not hasattr(settings, "DATASETS") or not settings.DATASETS:
        raise ValueError("Missing or empty 'DATASETS' in settings.")

    for engine, variants in settings.DYNAMIC_PIPELINES_MAPPING.items():
        engine_pipeline = build_engine_pipelines(
            engine, variants, start_match, last_match
        )
        pipeline_dict[engine] = etl_pipeline + engine_pipeline
        default_pipelines.append(engine_pipeline)

    pipeline_dict["__default__"] = etl_pipeline + pipeline(default_pipelines)

    return pipeline_dict
