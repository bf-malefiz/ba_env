"""
This pipeline is responsible for processing raw football match data into a structured format
suitable for machine learning. It performs the following steps:
1. Builds a team lexicon (mapping of team names to unique indices).
2. Extracts goal results and maps team names to their indices.
3. Vectorizes the goal results into a structured DataFrame.

The pipeline is designed to handle multiple datasets, as defined in the `settings.DATASETS` list.
Each dataset is processed independently, and the results are combined into a single pipeline.

Inputs:
- CSV: Raw match data in CSV format, containing columns like "HomeTeam", "AwayTeam", "FTHG", and "FTAG".

Outputs:
- team_lexicon: A DataFrame mapping team names to unique indices.
- goals: A DataFrame containing goal results with columns "home_goals" and "away_goals".
- vectorized_data: A structured DataFrame with columns "home_id", "away_id", "home_goals", "away_goals", and "toto".

Example:
    To create and run the pipeline:
    ```python
    from kedro.framework.project import pipelines
    pipeline = pipelines["__default__"]
    pipeline.run()
    ```
"""

from bundesliga import settings
from kedro.pipeline import Pipeline, node, pipeline

from .nodes_etl.nodes import (
    build_team_lexicon,
    get_goal_results,
    vectorize_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the data processing pipeline for football match data.

    This function defines a Kedro pipeline that processes raw match data into a structured format.
    The pipeline consists of three main nodes:
    1. `build_team_lexicon`: Creates a mapping of team names to unique indices.
    2. `get_goal_results`: Extracts goal results and maps team names to their indices.
    3. `vectorize_data`: Converts goal results into a structured DataFrame.

    The pipeline is designed to handle multiple datasets, as specified in the `settings.DATASETS` list.
    Each dataset is processed independently, and the results are combined into a single pipeline.

    Args:
        **kwargs: Additional keyword arguments (unused in this function).

    Returns:
        Pipeline: A Kedro pipeline object representing the data processing workflow.

    Example:
        To create and run the pipeline:
        ```python
        from kedro.framework.project import pipelines

        pipeline = pipelines["__default__"]
        pipeline.run()
        ```
    """
    # Define the base pipeline with three nodes
    data_processing = pipeline(
        pipe=[
            node(
                func=build_team_lexicon,
                inputs="CSV",
                outputs="team_lexicon",
                name="build_team_lexicon_node",
            ),
            node(
                func=get_goal_results,
                inputs=["CSV", "team_lexicon"],
                outputs="goals",
                name="get_goal_results_node",
                # namespace=None,
            ),
            node(
                func=vectorize_data,
                inputs="goals",
                outputs="vectorized_data",
                name="vectorize_data_node",
            ),
        ],
    )
    # Get the list of datasets from settings
    datasets_list = settings.DATASETS

    # Create a separate pipeline for each dataset
    dataset_pipelines = []
    for dataset_name in datasets_list:
        dataset_pipelines.append(
            pipeline(
                data_processing,
                namespace=dataset_name,  # Namespace for the dataset
                tags=["ETL", dataset_name],  # Tags for the dataset
            )
        )

    # Combine all dataset pipelines into a single pipeline
    return sum(dataset_pipelines)
