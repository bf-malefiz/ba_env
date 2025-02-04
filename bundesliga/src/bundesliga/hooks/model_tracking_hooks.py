from typing import Any, Dict

import mlflow
from bundesliga import settings
from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node


class ModelTrackingHooks:
    """
    A collection of Kedro hooks for tracking machine learning models using MLflow.

    This class implements various Kedro hooks to integrate MLflow tracking into the Kedro pipeline.
    It logs datasets, parameters, metrics, and tags at different stages of the pipeline execution.

    Attributes:
        None
    """


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
        if "evaluate" in node._func_name:
            # Get the evaluation results from the node outputs
            out_name = node._outputs
            eval_results = outputs[out_name]

            # Extract tags from the node (e.g., day, season, engine, model)
            tags = sorted(node.tags)
            day = tags[0]
            season = tags[1]
            engine = tags[2]
            model = tags[3]

            # Prepare the metrics for logging
            results = {
                "winner_accuracy": eval_results["winner_accuracy"],
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
                f"day: {day} | seed: {settings.SEED}"
            )

            # Log metrics and parameters to MLflow
            active_run = mlflow.active_run()
            if active_run is not None:
                with mlflow.start_run(run_name=run_name, nested=True) as run:
                    mlflow.log_params(
                        {
                            "day": day,
                            "season": season,
                            "model": model,
                            "engine": engine,
                            "run": "solo",
                            "seed": settings.SEED,
                            "run_id": run.info.run_id,
                        }
                    )

                    mlflow.log_metrics(results)
                    mlflow.set_tags(
                        {
                            "day": day,
                            "season": season,
                            "model": model,
                            "engine": engine,
                            "run": "solo",
                            "seed": settings.SEED,
                            "run_id": run.info.run_id,
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
        pass
