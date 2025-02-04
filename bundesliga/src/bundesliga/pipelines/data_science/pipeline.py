from bundesliga import settings
from bundesliga.utils.utils import split_time_data
from kedro.pipeline import Pipeline, node, pipeline

from .nodes_ml.evaluate import aggregate_eval_metrics
from .nodes_ml.train_model import (
    evaluate,
    init_model,
    predict_goals,
    train,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipes = []

    return pipeline(pipes)


def create_model_definition_node(engine, variant, dataset_name):
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


def eval_pipeline(startmatch, last_match, setting, dataset_name) -> Pipeline:
    """Erstellt eine Evaluationspipeline für verschiedene Engines und Varianten."""
    pipe_collection = []

    for engine, variants in setting:
        for variant in variants:
            metrics_inputs = [
                f"{engine}.{dataset_name}.{variant}.metrics_{match}"
                for match in range(startmatch, startmatch + last_match)
            ]

            pipe_collection.append(
                pipeline(
                    [
                        create_model_definition_node(engine, variant, dataset_name),
                        node(
                            func=aggregate_eval_metrics,
                            inputs=[
                                f"{engine}.{dataset_name}.{variant}.modeldefinitions"
                            ]
                            + metrics_inputs,
                            outputs=f"{engine}.{dataset_name}.{variant}.all_daily_metrics",
                            name=f"{engine}.{dataset_name}.{variant}.aggregate_eval_metrics_node",
                        ),
                    ]
                )
            )
    return pipeline(pipe_collection)


def ml_pipeline(start_match, last_match, engine, variant, dataset_name):
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
                func=evaluate,
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
