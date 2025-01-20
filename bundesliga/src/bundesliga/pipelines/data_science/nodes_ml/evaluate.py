import mlflow


def aggregate_eval_metrics(
    *kwargs,
):
    """
    all_daily_metrics: z. B. [{'winner_accuracy': 1.0, 'rmse_home': 1.2, ...}, {...}, ...]
    """
    model_definitions = kwargs[0]
    engine = model_definitions["engine"]
    dataset_name = model_definitions["dataset_name"]
    variant = model_definitions["variant"]
    seed = model_definitions["seed"]
    all_daily_metrics = kwargs[1:]

    accuracies = [
        m["winner_accuracy"] for m in all_daily_metrics if "winner_accuracy" in m
    ]
    if accuracies:
        avg_acc = sum(accuracies) / len(accuracies)
    else:
        avg_acc = 0.0

    nested_run_name = f"Aggregated Accuracy | engine={engine} | model={variant} | season={dataset_name}"

    with mlflow.start_run(run_name=nested_run_name, nested=True) as run:
        mlflow.log_metric("avg_winner_accuracy_over_all_days", avg_acc)
        mlflow.log_params(
            {
                "season": dataset_name,
                "model": variant,
                "engine": engine,
                "seed": seed,
                "run_id": run.info.run_id,
            }
        )
        mlflow.set_tags(
            {
                "model": variant,
                "engine": engine,
                "season": dataset_name,
                "seed": seed,
                "run_id": run.info.run_id,
            }
        )

    return {"avg_winner_accuracy_over_all_days": avg_acc}
