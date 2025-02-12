"""
Module: pipeline.py

Summary:
    This module defines Kedro pipelines for machine learning model training and evaluation
    for football match prediction. The pipelines include nodes for creating model definitions,
    training, prediction, evaluation, and aggregation of metrics across datasets and models.
    It supports a walk-forward evaluation approach over multiple matches.

Dependencies:
    - kedro.pipeline (Pipeline, node, pipeline)
    - bundesliga.settings
    - bundesliga.utils.utils (split_time_data)
    - .nodes_ml.evaluate (aggregate_dataset_metrics, aggregate_model_metrics, evaluate_match)
    - .nodes_ml.train_model (init_model, predict_goals, train)
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.node import Node

from bundesliga import settings
from bundesliga.utils.utils import split_time_data

from .nodes_ml.evaluate import (
    aggregate_dataset_metrics,
    aggregate_model_metrics,
    evaluate_match,
)
from .nodes_ml.train_model import (
    init_model,
    predict_goals,
    train,
)


def create_model_definition_node(
    engine: str, variant: str, dataset_name: str = None
) -> Node:
    """
    Creates a model definition node for eval_dataset_pipeline.

    This node generates a dictionary containing the model definition parameters such as the engine,
    variant, dataset name, and seed. The output of this node is used downstream to configure model
    training and evaluation nodes.

    Args:
        engine (str): The probabilistic programming engine (e.g., "pymc" or "pyro").
        variant (str): The model variant (e.g., "simple", "toto").
        dataset_name (str): The name of the dataset being processed. Defaults to None.

    Returns:
        Node: A dictionary with keys "engine", "variant", "dataset_name", and "seed".
    """
    return node(
        func=lambda: {
            "engine": engine,
            "variant": variant,
            "dataset_name": dataset_name,
            "seed": settings.SEED,
        },
        inputs=None,
        outputs=f"{engine}.{dataset_name}.{variant}.modeldefinitions",
        name=f"{engine}.{dataset_name}.{variant}.modeldef_node",
    )


def create_model_eval_definition_node(engine: str, variant: str) -> Node:
    """
    Creates a model evaluation definition node for eval_model_pipeline.

    This node generates a dictionary containing evaluation configuration parameters for a given engine
    and model variant, including the random seed. The output is used to aggregate evaluation metrics
    across datasets.

    Args:
        engine: The probabilistic programming engine (e.g., "pymc" or "pyro").
        variant: The model variant (e.g., "simple", "toto").

    Returns:
        dict: A dictionary with keys "engine", "variant", and "seed".
    """

    return node(
        func=lambda: {
            "engine": engine,
            "variant": variant,
            "seed": settings.SEED,
        },
        inputs=None,
        outputs=f"{engine}.{variant}.model_eval_definitions",
        name=f"{engine}.{variant}.model_eval_def_node",
    )


def eval_dataset_pipeline(
    start_match: int, last_match: int, setting: list[tuple[str, str]], dataset_name: str
) -> Pipeline:
    """
    Creates an evaluation pipeline for a dataset over a range of matches for various engines and variants.

    This pipeline aggregates evaluation metrics for individual matches within a dataset. For each combination
    of engine and model variant specified in the setting, it creates a subpipeline that collects metrics from
    each match and then aggregates them using the 'aggregate_dataset_metrics' node.

    Args:
        start_match (int): The starting match index for evaluation.
        last_match (int): The ending match index for evaluation.
        setting (list[tuple[str, str]]): An iterable of tuples specifying the engine and variant combinations.
        dataset_name (str): The name of the dataset to be processed.

    Returns:
        Pipeline: A Kedro pipeline object representing the aggregated dataset evaluation workflow.
    """
    pipe_collection = []

    for engine, variant in setting:
        match_metrics_inputs = [
            f"{engine}.{dataset_name}.{variant}.metrics_{match}"
            for match in range(start_match, last_match)
        ]

        pipe_collection.append(
            pipeline(
                [
                    create_model_definition_node(engine, variant, dataset_name),
                    node(
                        func=aggregate_dataset_metrics,
                        inputs=[f"{engine}.{dataset_name}.{variant}.modeldefinitions"]
                        + match_metrics_inputs,
                        outputs=[
                            f"{engine}.{dataset_name}.{variant}.mean_metrics",
                            f"{engine}.{dataset_name}.{variant}.nested_run_name",
                            f"{engine}.{dataset_name}.{variant}.matchday_metrics",
                        ],
                        name=f"{engine}.{dataset_name}.{variant}.aggregate_dataset_metrics_node",
                        tags=[engine, variant, dataset_name],
                    ),
                ]
            )
        )

    return pipeline(pipe_collection)


def eval_model_pipeline(engine: str, variant: str) -> Pipeline:
    """
    Creates an evaluation pipeline to aggregate model metrics across datasets.

    This pipeline collects dataset-level metrics for a given engine and model variant from all datasets
    specified in the settings, then aggregates these metrics using the 'aggregate_model_metrics' node.

    Args:
        engine (str): The probabilistic programming engine (e.g., "pymc" or "pyro").
        variant (str): The model variant (e.g., "simple", "toto").

    Returns:
        Pipeline: A Kedro pipeline object representing the model evaluation aggregation workflow.
    """
    match_metrics_inputs = [
        f"{engine}.{dataset_name}.{variant}.mean_metrics"
        for dataset_name in settings.DATASETS
    ]
    pipe_collection = []
    pipe_collection.append(
        pipeline(
            [
                create_model_eval_definition_node(engine, variant),
                node(
                    func=aggregate_model_metrics,
                    inputs=[
                        f"{engine}.{variant}.model_eval_definitions",
                    ]
                    + match_metrics_inputs,
                    outputs=[
                        f"{engine}.{variant}.model_metrics",
                        f"{engine}.{variant}.nested_run_name",
                    ],
                    name=f"{engine}.{variant}.model_metrics_node",
                    tags=[engine, variant],
                ),
            ]
        )
    )
    return pipeline(pipe_collection)


def ml_pipeline(
    start_match: int, last_match: int, variant: str, dataset_name: str
) -> Pipeline:
    """
    Creates a machine learning pipeline for training and evaluating a model using a walk-forward approach.

    This pipeline integrates the walk-forward pipeline with input datasets and model configuration parameters.
    It processes the team lexicon and vectorized data, and routes them to the appropriate nodes for model
    initialization, training, prediction, and evaluation.

    Args:
        start_match (int): The starting match index for the walk-forward evaluation.
        last_match (int): The ending match index for the walk-forward evaluation.
        variant (str): The model variant to be used (e.g., "simple", "toto").
        dataset_name (str): The name of the dataset being processed.

    Returns:
        Pipeline: A Kedro pipeline object representing the machine learning workflow.
    """
    return pipeline(
        create_walk_forward_pipeline(start_match, last_match),
        inputs={
            "team_lexicon": "team_lexicon",
            "vectorized_data": "vectorized_data",
        },
        parameters={"params:model_options": f"params:{variant}.model_options"},
        namespace=f"{dataset_name}.{variant}",
        tags=[variant, dataset_name],
    )


def create_walk_forward_pipeline(start_match: int, last_match: int) -> Pipeline:
    """
    Creates a walk-forward pipeline that iterates over a sequence of matches.

    This function constructs a pipeline that sequentially processes matches from start_match to last_match.
    It combines subpipelines for individual matches using the sum operator, resulting in a unified pipeline
    that applies the walk-forward evaluation approach.

    Args:
        start_match (int): The index of the first match to process.
        last_match (int): The index of the last match to process.

    Returns:
        Pipeline: A Kedro pipeline object representing the walk-forward evaluation workflow.
    """
    return sum(
        (
            create_subpipeline_for_match(match)
            for match in range(start_match, last_match)
        ),
        start=Pipeline([]),
    )


def create_subpipeline_for_match(match: int) -> Pipeline:
    """
    Creates a subpipeline for processing a single match.

    The subpipeline includes nodes for:
      - Identifying the match.
      - Splitting the vectorized data into training and testing sets.
      - Initializing the model.
      - Training the model.
      - Predicting goals.
      - Evaluating the model's performance.

    Args:
        match (int): The match index for which the subpipeline is created.

    Returns:
        Pipeline: A Kedro pipeline object representing the subworkflow for a single match.
    """
    return Pipeline(
        [
            node(
                func=lambda: str(match),
                inputs=None,
                outputs=f"match_{match}",
                name=f"match_node_{match}",
            ),
            node(
                func=split_time_data,
                inputs={
                    "vectorized_data": "vectorized_data",
                    "current_match": f"match_{match}",
                },
                outputs=[
                    f"train_data_{match}",
                    f"test_data_{match}",
                ],
                name=f"split_node_{match}",
            ),
            node(
                func=init_model,
                inputs={
                    "team_lexicon": "team_lexicon",
                    "model_options": "params:model_options",
                    # "toto": f"toto_{match}",
                },
                outputs=f"init_model_{match}",
                name=f"init_model_node_{match}",
            ),
            node(
                func=train,
                inputs={
                    "model": f"init_model_{match}",
                    "train_data": f"train_data_{match}",
                    "model_options": "params:model_options",
                },
                outputs=f"model_{match}",
                name=f"fit_node_{match}",
            ),
            node(
                func=predict_goals,
                inputs={
                    "model": f"model_{match}",
                    "test_data": f"test_data_{match}",
                },
                outputs=f"predictions_{match}",
                name=f"predict_node_{match}",
            ),
            node(
                func=evaluate_match,
                inputs={
                    "model": f"model_{match}",
                    "test_data": f"test_data_{match}",
                    "predictions": f"predictions_{match}",
                },
                outputs=f"metrics_{match}",
                name=f"evaluate_node_{match}",
            ),
        ],
        tags=[f"{match}"],
    )
