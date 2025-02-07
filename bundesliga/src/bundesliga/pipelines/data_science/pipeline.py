from kedro.pipeline import Pipeline, node, pipeline

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


def create_model_definition_node(engine, variant, dataset_name=None):
    """Erstellt eine Modelldefinitions-Node für eine Pipeline."""
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


def create_model_eval_definition_node(engine, variant):
    """Erstellt eine Modelldefinitions-Node für eine Pipeline."""
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


def eval_dataset_pipeline(start_match, last_match, setting, dataset_name) -> Pipeline:
    """Erstellt eine Evaluationspipeline für verschiedene Engines und Varianten."""
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
                            f"{engine}.{dataset_name}.{variant}.dataset_metrics",
                            f"{engine}.{dataset_name}.{variant}.nested_run_name",
                        ],
                        name=f"{engine}.{dataset_name}.{variant}.aggregate_dataset_metrics_node",
                        tags=[engine, variant, dataset_name],
                    ),
                ]
            )
        )

    return pipeline(pipe_collection)


def eval_model_pipeline(engine, variant):
    match_metrics_inputs = [
        f"{engine}.{dataset_name}.{variant}.dataset_metrics"
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


def ml_pipeline(start_match, last_match, variant, dataset_name):
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
    Erstellt eine Pipeline, die für mehrere Tage durchlaufen wird.
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
    Erzeugt Knoten [match -> split -> init -> train -> predict -> evaluate].
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
                    # "toto": f"toto_{match}",
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
                    "match": f"match_{match}",
                    "test_data": f"test_data_{match}",
                    "predictions": f"predictions_{match}",
                },
                outputs=f"metrics_{match}",
                name=f"evaluate_node_{match}",
            ),
        ],
        tags=[f"{match}"],
    )
