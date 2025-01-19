from typing import Any, Dict

import mlflow
from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node


class ModelTrackingHooks:
    """Namespace for grouping all model-tracking hooks with MLflow together."""

    @hook_impl
    def after_context_created(self, context):
        pass

    @hook_impl
    def after_dataset_loaded(self, dataset_name, data, node):
        pass

    @hook_impl
    def before_pipeline_run(
        self, run_params: Dict[str, Any], pipeline, catalog
    ) -> None:
        """Hook implementation to start an MLflow run
        with the same run_id as the Kedro pipeline run.
        """

        node_names = set([node.short_name for node in pipeline.nodes])
        run_params["node_names"] = node_names
        mlflow.log_params(run_params)

    @hook_impl
    def before_node_run(self, node: Node, inputs: Dict[str, Any]) -> None:
        """Hook implementation to log the parameters before some node runs.
        In this example, we will log the parameters before the data splitting node runs.
        """
        pass

    @hook_impl
    def after_node_run(
        self, node: Node, outputs: Dict[str, Any], inputs: Dict[str, Any]
    ) -> None:
        if "evaluate" in node._func_name:
            # Hol dir das (Dictionary) mit den Kennzahlen aus den Node-Outputs
            out_name = node._outputs
            eval_results = outputs[out_name]

            # Sortierte Tags -> meist Tag-Reihenfolge wie [day, season, engine, model]
            tags = sorted(node.tags)

            # Wenn du weißt, welche Indexe was bedeuten, kannst du es hier namentlich auseinandernehmen
            day = tags[0]
            season = tags[1]
            engine = tags[2]
            model = tags[3]

            # Optional: Du kannst auch `.index()` auf die Tags machen,
            # um die Position zu finden oder Tag-Strings direkt suchen.

            # Bereite die Metriken auf
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

            # Run-Name (z.B. "model: poisson Season: 2023 day: 2"):
            run_name = (
                f"engine: {engine} | model: {model} | season: {season} | day: {day}"
            )

            active_run = mlflow.active_run()
            if active_run is not None:
                # Falls wir nested logging wollen
                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.log_params(
                        {
                            "day": day,
                            "season": season,
                            "model": model,
                            "engine": engine,
                        }
                    )
                    mlflow.log_metrics(results)
                    mlflow.set_tags(
                        {
                            "day": day,
                            "season": season,
                            "model": model,
                            "engine": engine,
                        }
                    )
            else:
                # Falls wider Erwarten kein aktiver Run da ist, kannst du
                # selber einen starten, oder einfach loggen.
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params(
                        {
                            "day": day,
                            "season": season,
                            "model": model,
                            "engine": engine,
                        }
                    )
                    mlflow.log_metrics(results)
                    mlflow.set_tags(
                        {
                            "day": day,
                            "season": season,
                            "model": model,
                            "engine": engine,
                        }
                    )

    @hook_impl
    def after_pipeline_run(self, run_params, run_result, pipeline, catalog) -> None:
        pass
