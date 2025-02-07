from typing import Any, Dict

import mlflow
from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node

from bundesliga import settings


class ModelTrackingHooks:
    """
    A collection of Kedro hooks for tracking machine learning models using MLflow.

    This class implements various Kedro hooks to integrate MLflow tracking into the Kedro pipeline.
    It logs datasets, parameters, metrics, and tags at different stages of the pipeline execution.

    Attributes:
        None
    """

    def __init__(self):
        self.parent_run = None

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        run = mlflow.active_run()
        mlflow.end_run()
        mlflow.delete_run(run.info.run_id)

    @hook_impl
    def after_dataset_loaded(self, dataset_name, data, node):
        """
        Hook implementation called after a dataset is loaded.

        This hook logs the dataset to MLflow if the dataset name contains "vectorize".

        Args:
            dataset_name (str): The name of the dataset that was loaded.
            data (Any): The data that was loaded.
            node (Node): The Kedro node that loaded the dataset.
        """
        pipeline_tag = ""

        if "__default__" in node.tags:
            pipeline_tag = "__default__"
        else:
            pipeline_tag = next(
                (
                    tag
                    for tag in node.tags
                    if tag in settings.DYNAMIC_PIPELINES_MAPPING.keys()
                ),
                None,
            )
        run = mlflow.active_run()
        if run is not None and run.info.run_name != pipeline_tag:
            mlflow.end_run()
            mlflow.delete_run(run.info.run_id)
            mlflow.start_run(
                run_name=pipeline_tag,
                description=f"Main Run for CLI instantiated pipeline: {pipeline_tag}",
            )

        if "vectorize" in dataset_name:
            pd_dataset = mlflow.data.from_pandas(data, name=dataset_name)

            mlflow.log_input(pd_dataset, context=dataset_name)

    @hook_impl
    def before_node_run(self, node: Node, inputs: Dict[str, Any]) -> None:
        """
        Hook implementation called before a node runs.

        This hook can be used to log parameters before a node runs. In this example, it is not implemented.

        Args:
            node (Node): The Kedro node about to be executed.
            inputs (Dict[str, Any]): The inputs to the node.
        """
        pass

    @hook_impl
    def after_node_run(
        self, node: Node, outputs: Dict[str, Any], inputs: Dict[str, Any]
    ) -> None:
        """
        Hook implementation called after a node runs.

        This hook logs evaluation metrics to MLflow if the node's function name contains "evaluate".

        Args:
            node (Node): The Kedro node that was executed.
            outputs (Dict[str, Any]): The outputs from the node.
            inputs (Dict[str, Any]): The inputs to the node.
        """

        tags = sorted(node.tags)

        if "evaluate" in node._func_name:
            match = tags[0]
            season = tags[1]
            run_arg = tags[2]
            engine = tags[3]
            model = tags[4]
            # Get the evaluation results from the node outputs
            out_name = node._outputs
            eval_results = outputs[out_name]

            # Prepare the metrics for logging
            results = {
                "home_prob": eval_results["home_prob"],
                "away_prob": eval_results["away_prob"],
                "tie_prob": eval_results["tie_prob"],
                "rmse_home": eval_results["rmse_home"],
                "mae_home": eval_results["mae_home"],
                "rmse_away": eval_results["rmse_away"],
                "mae_away": eval_results["mae_away"],
                "neg_log_likelihood": eval_results["neg_log_likelihood"],
                "brier_score": eval_results["brier_score"],
                "rps": eval_results["rps"],
            }

            # Create a run name for MLflow
            run_name = (
                f"Solorun - engine: {engine} | model: {model} | season: {season} | "
                f"match: {match} | seed: {settings.SEED}"
            )

            # Log metrics and parameters to MLflow
            active_run = mlflow.active_run()
            if active_run is not None:
                with mlflow.start_run(
                    run_name=run_name,
                    nested=True,
                ) as run:
                    mlflow.log_params(
                        {
                            "match": match,
                            "season": season,
                            "model": model,
                            "engine": engine,
                            "run": "solo",
                            "seed": settings.SEED,
                            "run_id": run.info.run_id,
                            "ground_truth": eval_results["ground_truth"],
                            "predicted_result": eval_results["predicted_result"],
                        }
                    )

                    mlflow.log_metrics(results)
                    mlflow.set_tags(
                        {
                            "match": match,
                            "season": season,
                            "model": model,
                            "engine": engine,
                            "run": "solo",
                            "seed": settings.SEED,
                            "run_id": run.info.run_id,
                        }
                    )
        if "aggregate_dataset_metrics" in node._func_name:
            # Get the evaluation results from the node outputs
            out_names = node._outputs
            mean_metrics = outputs[out_names[0]]
            nested_run_name = outputs[out_names[1]]

            # Log metrics and parameters to MLflow
            active_run = mlflow.active_run()
            if active_run is not None:
                with mlflow.start_run(
                    # parent_run_id=self.parent_run.info.run_id,
                    run_name=nested_run_name,
                    nested=True,
                ) as run:
                    mlflow.log_metrics(mean_metrics)
                    mlflow.log_params(
                        {
                            "season": tags[0],
                            "pipeline": tags[1],
                            "engine": tags[2],
                            "model": tags[3],
                            "seed": settings.SEED,
                        }
                    )
        if "aggregate_model_metrics" in node._func_name:
            # Get the evaluation results from the node outputs
            out_names = node._outputs
            mean_metrics = outputs[out_names[0]]
            nested_run_name = outputs[out_names[1]]

            # Log metrics and parameters to MLflow
            active_run = mlflow.active_run()
            if active_run is not None:
                with mlflow.start_run(
                    # parent_run_id=self.parent_run.info.run_id,
                    run_name=nested_run_name,
                    nested=True,
                ) as run:
                    mlflow.log_metrics(mean_metrics)
                    mlflow.log_params(
                        {
                            "pipeline": tags[0],
                            "engine": tags[1],
                            "model": tags[2],
                            "seed": settings.SEED,
                        }
                    )

    @hook_impl
    def after_pipeline_run(self, run_params, run_result, pipeline, catalog) -> None:
        """
        Hook implementation called after a pipeline run completes.

        This hook can be used to perform cleanup tasks or log final results after the pipeline run.

        Args:
            run_params: Parameters for the pipeline run.
            run_result: The result of the pipeline run.
            pipeline: The Kedro pipeline that was executed.
            catalog: The Kedro data catalog.
        """
