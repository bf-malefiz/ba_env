import mlflow
import pandas as pd
from bundesliga.utils.utils import brier_score, log_likelihood, rmse_mae, rps
from bundesliga.utils.validation import validate_dataframe


@validate_dataframe(
    df_arg_name="test_data",
    required_columns=["home_goals", "away_goals"],
    allow_empty=False,
)
@validate_dataframe(
    df_arg_name="predictions",
    required_columns=["home_goals", "away_goals"],
    allow_empty=False,
)
def evaluate_match(model, match, test_data, predictions):
    test_goal = test_data[["home_goals", "away_goals"]]
    test_goal = test_goal.copy()
    test_goal["true_result"] = test_goal.apply(true_result, axis=1)

    toto_probs = model.predict_toto_probabilities(predictions)
    toto_probs = pd.DataFrame(toto_probs, columns=["home", "away", "tie"])
    toto_probs["predicted_result"] = toto_probs.apply(predicted_result, axis=1)

    test_goal = test_goal.reset_index(drop=True)
    toto_probs = toto_probs.reset_index(drop=True)

    # winner_accuracy = (
    #     test_goal["true_result"] == toto_probs["predicted_result"]
    # ).mean()

    negative_log_likelihood = log_likelihood(toto_probs, test_goal)
    brier_score_ = brier_score(test_goal, toto_probs)
    rps_mean = rps(test_goal, toto_probs)
    rmse_home, mae_home, rmse_away, mae_away = rmse_mae(predictions, test_data)

    predictions["predicted_result"] = determine_winner(predictions)["winner"]
    toto_probs["ground_truth"] = test_goal["true_result"].values

    ground_truth_met = (
        test_goal["true_result"].values == predictions["predicted_result"].values
    )
    winner_accuracy = (ground_truth_met).mean()

    results = {
        "winner_accuracy": winner_accuracy,
        "toto_probs": toto_probs,
        "rmse_home": rmse_home,
        "mae_home": mae_home,
        "rmse_away": rmse_away,
        "mae_away": mae_away,
        "neg_log_likelihood": negative_log_likelihood,
        "brier_score": brier_score_,
        "rps": rps_mean,
    }
    return results


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


@validate_dataframe(
    df_arg_name="predictions",
    required_columns=["home_goals", "away_goals"],
    allow_empty=False,
)
def determine_winner(predictions):
    # Funktion, um den Gewinner zu bestimmen
    def get_winner(row):
        if row["home_goals"] > row["away_goals"]:
            return "home"
        elif row["home_goals"] < row["away_goals"]:
            return "away"
        else:
            return "tie"

    # Anwenden der Funktion auf jede Zeile des DataFrames
    predictions["winner"] = predictions.apply(get_winner, axis=1)
    return predictions


def true_result(df_row):
    if df_row["home_goals"] > df_row["away_goals"]:
        return "home"
    elif df_row["home_goals"] < df_row["away_goals"]:
        return "away"
    else:
        return "tie"


def predicted_result(probs_row):
    # probs_row = { "home": ..., "away": ..., "tie": ... }
    return probs_row.idxmax()
