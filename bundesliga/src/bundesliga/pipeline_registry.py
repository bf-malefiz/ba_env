"""Project pipelines."""

from kedro.pipeline import Pipeline, pipeline
from kedro_datasets.pandas import CSVDataset

from bundesliga import settings
from bundesliga.pipelines.data_processing.pipeline import (
    create_pipeline as create_etl_pipeline,
)
from bundesliga.pipelines.data_science.pipeline import (
    eval_dataset_pipeline,
    eval_model_pipeline,
    ml_pipeline,
)
from bundesliga.utils.utils import load_config


def read_matchlength_from_data(dataset_name: str, start_match: int) -> int:
    """
    Reads the walk-forward parameter (match length) from the raw data catalog.

    This function loads raw match data for the given dataset from a CSV file, determines the total number
    of matches, and calculates the effective number of matches available for walk-forward evaluation.
    It ensures that the provided start_match is within valid bounds.

    Args:
        dataset_name (str): The name of the dataset.
        start_match (int): The starting match index for evaluation.

    Returns:
        int: The number of matches available for walk-forward evaluation, computed as total_matches - start_match - 1.

    Raises:
        ValueError: If the start_match is greater than or equal to the total number of matches in the dataset.
    """
    data = CSVDataset(
        filepath=f"./data/01_raw/football-datasets/{dataset_name}.csv",
        load_args=dict(sep=",", encoding="cp1252", index_col=0),
    )
    df = data.load()
    total_matches = len(df)
    if start_match >= total_matches:
        raise ValueError(
            f"Start match {start_match} is greater than or equal the number of matches in the dataset {len(df)}."
        )
    return total_matches - start_match - 1


def build_engine_pipelines(
    engine: str, variants: list[str], start_match: int = 0, last_match: int = None
) -> Pipeline:
    """
    Builds a collection of machine learning pipelines for a specific engine and its variants.

    This function creates a pipeline for each combination of the specified engine, model variant, and dataset.
    It combines the machine learning pipeline (ml_pipeline) with the evaluation pipeline (eval_dataset_pipeline)
    for each dataset defined in the project settings, and then aggregates them with the model evaluation pipeline.
    This allows for a complete workflow of model training, prediction, and evaluation for each engine and variant.

    Args:
        engine (str): The engine type (e.g., "pymc", "pyro").
        variants (list[str]): A list of model variants for the engine.
        start_match (int): The starting match index for processing.
        last_match (int): The ending match index for processing.

    Returns:
        Pipeline: A Kedro pipeline object representing the combined machine learning and evaluation workflows
                  for the specified engine and variants.

    Raises:
        ValueError: If required parameters or settings (e.g., engine, variants, DATASETS) are missing or invalid.
    """

    if not engine:
        raise ValueError("Engine type cannot be empty.")

    if not variants:
        raise ValueError("Variants list cannot be empty.")

    if not hasattr(settings, "DATASETS") or not settings.DATASETS:
        raise ValueError("Missing or empty 'DATASETS' in settings.")

    engine_pipelines = []

    for variant in variants:
        for dataset_name in settings.DATASETS:
            if not last_match:
                last_match = read_matchlength_from_data(
                    dataset_name, start_match=start_match
                )
            try:
                pipeline_obj = pipeline(
                    ml_pipeline(
                        start_match=start_match,
                        last_match=last_match,
                        variant=variant,
                        dataset_name=dataset_name,
                    ),
                    inputs={
                        "team_lexicon": f"{dataset_name}.team_lexicon",
                        "vectorized_data": f"{dataset_name}.vectorized_data",
                    },
                    namespace=f"{engine}",
                    tags=[engine],
                ) + eval_dataset_pipeline(
                    start_match=start_match,
                    last_match=last_match,
                    setting=[(engine, variant)],
                    dataset_name=dataset_name,
                )
                engine_pipelines.append(pipeline_obj)

            except Exception as e:
                raise ValueError(
                    f"Failed to build pipeline for engine '{engine}', variant '{variant}', "
                    f"and dataset '{dataset_name}': {str(e)}"
                )
            if not last_match:
                last_match = None
        engine_pipelines.append(eval_model_pipeline(engine, variant))
    return pipeline(engine_pipelines)


def register_pipelines() -> dict[str, Pipeline]:
    """
    Registers and constructs the project's pipelines for the Kedro process.

    This function dynamically creates and registers multiple pipelines based on the project configuration.
    It combines the ETL pipeline with machine learning and evaluation pipelines for each engine and model variant.
    The resulting pipelines are organized into a dictionary with keys such as "etl", specific engine names,
    and a default pipeline "__default__" that encompasses all workflows.

    Returns:
        dict[str, Pipeline]: A dictionary mapping pipeline names to their corresponding Kedro Pipeline objects.

    Raises:
        ValueError: If required parameters (e.g., start_match) or settings (e.g., DYNAMIC_PIPELINES_MAPPING, DATASETS)
                    are missing or invalid.
    """

    # Load configuration parameters
    parameters = load_config()["parameters"]
    start_match = parameters["model_options"]["start_match"]
    last_match = parameters["model_options"]["last_match"]

    # Create the ETL pipeline and initialize the pipeline dictionary
    etl_pipeline = create_etl_pipeline()
    pipeline_dict = {"etl": etl_pipeline}
    default_pipelines = Pipeline([])

    # Validate required parameters and settings
    if start_match is None:
        raise ValueError("Missing required parameters: 'start_match'.")
    if (
        not hasattr(settings, "DYNAMIC_PIPELINES_MAPPING")
        or not settings.DYNAMIC_PIPELINES_MAPPING
    ):
        raise ValueError("Missing or empty 'DYNAMIC_PIPELINES_MAPPING' in settings.")

    if not hasattr(settings, "DATASETS") or not settings.DATASETS:
        raise ValueError("Missing or empty 'DATASETS' in settings.")

    # Build pipelines for each engine and its variants
    for engine, variants in settings.DYNAMIC_PIPELINES_MAPPING.items():
        engine_pipeline = build_engine_pipelines(
            engine, variants, start_match, last_match
        )

        # Combine the ETL pipeline with the engine-specific pipelines
        combined_pipeline = etl_pipeline + pipeline(engine_pipeline)
        pipeline_dict[engine] = pipeline(
            combined_pipeline, tags={"pipeline_name": "_" + engine}
        )
        # Aggregate all engine pipelines for the default pipeline
        default_pipelines += engine_pipeline

    # Combine default pipelines with the ETL pipeline and register as the default pipeline
    default_pipeline = default_pipelines + etl_pipeline
    pipeline_dict["__default__"] = pipeline(
        default_pipeline, tags={"pipeline_name": "__default__"}
    )
    return pipeline_dict
