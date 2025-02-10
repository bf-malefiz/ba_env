from typing import Any

import mlflow
from kedro.framework.hooks import hook_impl
from kedro.io import CatalogProtocol
from kedro.pipeline import Pipeline
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
    def before_pipeline_run(
        self, run_params: (dict[str, Any]), pipeline: Pipeline, catalog: CatalogProtocol
    ) -> None:
        """
        Hook implementation called before the pipeline starts running.

        This method is designed to ensure that any previously active MLflow run is properly terminated
        before a new pipeline execution begins. It checks for an active MLflow run, and if one is found,
        it ends and deletes the run to prevent interference with the new pipeline run's tracking information.

        This method becomes opsolete when https://github.com/Galileo-Galilei/kedro-mlflow/issues/623 is resolved.

        Args:
            run_params (dict[str, Any]): A collection of parameters and settings associated with the pipeline run.
            pipeline (Pipeline): The Kedro pipeline object that is scheduled to run, containing the defined nodes.
            catalog (CatalogProtocol): The data catalog that provides configuration details for all datasets used in the pipeline.
        """
        run = mlflow.active_run()
        mlflow.end_run()
        mlflow.delete_run(run.info.run_id)

    @hook_impl
    def after_dataset_loaded(self, dataset_name: str, data: Any, node: Node) -> None:
        """
        Hook implementation called after a dataset is loaded.

        This method performs post-processing once a dataset is loaded into the pipeline. It inspects the node's
        tags to determine the appropriate pipeline context. If the current active MLflow run's name does not match
        the determined pipeline tag, the method ends and deletes the current run, then starts a new MLflow run
        with the proper run name and description. Additionally, if the dataset name contains the substring "vectorize",
        the dataset (assumed to be in a pandas DataFrame format) is converted into an MLflow dataset object and logged
        as an input for MLflow tracking.

        Args:
            dataset_name (str): The identifier of the dataset that was loaded, which may include hints (like "vectorize")
                                for further processing.
            data (Any): The actual dataset loaded which will be logged if it meets the criteria.
            node (Node): The Kedro node responsible for loading the dataset, whose tags are used to determine the appropriate pipeline context.
        """
        from bundesliga import settings

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
    def after_node_run(
        self, node: Node, outputs: dict[str, Any], inputs: dict[str, Any]
    ) -> None:
        from bundesliga import settings

        """
        Hook implementation called after a node completes its execution.

        This method handles post-execution tracking for different types of nodes based on the node function name.
        If the node's function name includes "evaluate", the method extracts specific tags that denote match details,
        season, engine, and model, then logs detailed evaluation metrics (such as probabilities, errors, and likelihoods)
        to MLflow in a nested run. Additionally, for nodes with function names including "aggregate_dataset_metrics" or
        "aggregate_model_metrics", the method logs aggregated mean metrics along with relevant parameters such as season,
        pipeline, engine, model, and a predefined seed. This enables a comprehensive tracking of both individual and
        aggregated performance metrics in MLflow for post-run analysis.

        Args:
            node (Node): The Kedro node that has executed, carrying attributes (like _func_name and tags) used for logging.
            outputs (Dict[str, Any]): The outputs produced by the node.
            inputs (Dict[str, Any]): The inputs provided to the node during its execution.
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
                        }
                    )

                    mlflow.log_metrics(eval_results)
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
            ds_mean_metrics = outputs[out_names[0]]
            nested_run_name = outputs[out_names[1]]

            # Log metrics and parameters to MLflow
            active_run = mlflow.active_run()
            if active_run is not None:
                with mlflow.start_run(
                    run_name=nested_run_name,
                    nested=True,
                ) as run:
                    mlflow.log_metrics(ds_mean_metrics)
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
            model_mean_metrics = outputs[out_names[0]]
            nested_run_name = outputs[out_names[1]]

            # Log metrics and parameters to MLflow
            active_run = mlflow.active_run()
            if active_run is not None:
                with mlflow.start_run(
                    run_name=nested_run_name,
                    nested=True,
                ) as run:
                    mlflow.log_metrics(model_mean_metrics)
                    mlflow.log_params(
                        {
                            "pipeline": tags[0],
                            "engine": tags[1],
                            "model": tags[2],
                            "seed": settings.SEED,
                        }
                    )
