"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""

from bundesliga import settings
from kedro.pipeline import Pipeline, node, pipeline
from utils import split_time_data

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


def eval_pipeline(
    startday,
    walks,
    setting,
    dataset_name,
) -> Pipeline:
    # 2) Erzeuge eine Node, die alle metrics_{day} einließt

    pipe_collection = []

    for engine, variants in setting:
        for variant in variants:
            metrics_inputs = []
            for day in range(startday, startday + walks):
                metrics_inputs.append(
                    f"{engine}.{dataset_name}.{variant}.metrics_{day}"
                )

            pipe_collection.append(
                pipeline(
                    [
                        node(
                            func=lambda e=engine, v=variant, ds=dataset_name: {
                                "engine": e,
                                "variant": v,
                                "dataset_name": ds,
                                "seed": settings.SEED,
                            },
                            inputs=None,
                            outputs=f"{engine}.{dataset_name}.{variant}.modeldefinitions",
                            name=f"{engine}.{dataset_name}.{variant}.modeldef_node",
                        ),
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


def ml_pipeline(start_day, walk_forward, engine, variant, dataset_name):
    return pipeline(
        create_walk_forward_pipeline(start_day, walk_forward),
        inputs={
            "team_lexicon": "team_lexicon",
            "vectorized_data": "vectorized_data",
        },
        parameters={"params:model_options": f"params:{variant}.model_options"},
        namespace=f"{dataset_name}.{variant}",
        tags=[variant, dataset_name],
    )


def create_walk_forward_pipeline(start_day: int, times_to_walk: int) -> Pipeline:
    """
    Baut eine Pipeline, die für day=1..max_day Sub-Pipelines kaskadiert.
    => Ein einziger Pipeline-Lauf deckt alle Tage ab.
    """
    subpipelines = []
    for day in range(start_day, start_day + times_to_walk + 1):
        subpipelines.append(create_subpipeline_for_day(day))

    # # Sub-Pipelines zu einer großen Pipeline zusammenfassen
    # walk_forward_pipeline_collection = reduce(lambda p1, p2: p1 + p2, subpipelines)
    return subpipelines


def create_subpipeline_for_day(day: int) -> Pipeline:
    """
    Erzeugt Knoten [split -> fit -> predict -> evaluate] für Tag = `day`.
    """
    return Pipeline(
        [
            node(
                func=lambda: str(day),
                inputs=None,
                outputs=f"day_{day}",
                name=f"day_node_{day}",
            ),
            node(
                func=split_time_data,
                inputs={
                    "vectorized_data": "vectorized_data",
                    "current_day": f"day_{day}",
                },
                outputs=[
                    f"train_data_{day}",
                    f"test_data_{day}",
                ],
                name=f"split_node_{day}",
            ),
            node(
                func=init_model,
                inputs={
                    "team_lexicon": "team_lexicon",
                    "parameters": "params:model_options",
                },
                outputs=f"init_model_{day}",
                name=f"init_model_node_{day}",
            ),
            node(
                func=train,
                inputs={
                    "model": f"init_model_{day}",
                    "train_data": f"train_data_{day}",
                    "parameters": "params:model_options",
                    # "toto": f"toto_{day}",
                },
                outputs=[f"model_{day}", f"idata_{day}"],
                name=f"fit_node_{day}",
            ),
            node(
                func=predict_goals,
                inputs={
                    "model": f"model_{day}",
                    "test_data": f"test_data_{day}",
                    "parameters": "params:model_options",
                },
                outputs=f"predictions_{day}",
                name=f"predict_node_{day}",
            ),
            node(
                func=evaluate,
                inputs={
                    "model": f"model_{day}",
                    "day": f"day_{day}",
                    "test_data": f"test_data_{day}",
                    "predictions": f"predictions_{day}",
                },
                outputs=f"metrics_{day}",
                name=f"evaluate_node_{day}",
            ),
        ],
        tags=[f"{day}"],
    )
