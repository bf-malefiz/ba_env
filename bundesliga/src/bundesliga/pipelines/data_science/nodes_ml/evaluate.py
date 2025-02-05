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

    # Additional evaluation metric examples
    negative_log_likelihood = log_likelihood(toto_probs, test_goal)
    brier_score_ = brier_score(test_goal, toto_probs)
    rps_mean = rps(test_goal, toto_probs)
    rmse_home, mae_home, rmse_away, mae_away = rmse_mae(predictions, test_data)

    results = {
        "ground_truth": test_goal["true_result"].values[0],
        "predicted_result": toto_probs["predicted_result"].values[0],
        "home_prob": toto_probs["home"].values[0],
        "away_prob": toto_probs["away"].values[0],
        "tie_prob": toto_probs["tie"].values[0],
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


def aggregate_dataset_metrics(
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
    all_daily_metrics = pd.DataFrame(kwargs[1:])

    nested_run_name = f"Dataset Metrics | engine={engine} | model={variant} | season={dataset_name}"

    all_daily_metrics["correct"] = all_daily_metrics.apply(
        lambda row: row["predicted_result"] == row["ground_truth"], axis=1
    )

    # Now compute the mean for each metric. You can either compute them individually:
    mean_accuracy = all_daily_metrics["correct"].mean()
    mean_rmse_home = all_daily_metrics["rmse_home"].mean()
    mean_mae_home = all_daily_metrics["mae_home"].mean()
    mean_rmse_away = all_daily_metrics["rmse_away"].mean()
    mean_mae_away = all_daily_metrics["mae_away"].mean()
    mean_neg_log_likelihood = all_daily_metrics["neg_log_likelihood"].mean()
    mean_brier_score = all_daily_metrics["brier_score"].mean()
    mean_rps = all_daily_metrics["rps"].mean()

    mean_metrics = {
        "accuracy": mean_accuracy,
        "rmse_home": mean_rmse_home,
        "mae_home": mean_mae_home,
        "rmse_away": mean_rmse_away,
        "mae_away": mean_mae_away,
        "neg_log_likelihood": mean_neg_log_likelihood,
        "brier_score": mean_brier_score,
        "rps": mean_rps,
    }

    return mean_metrics, nested_run_name


def aggregate_model_metrics(
    *kwargs,
):
    """
    all_daily_metrics: z. B. [{'winner_accuracy': 1.0, 'rmse_home': 1.2, ...}, {...}, ...]
    """

    model_definitions = kwargs[0]
    engine = model_definitions["engine"]
    variant = model_definitions["variant"]
    seed = model_definitions["seed"]
    all_dataset_metrics = pd.DataFrame(kwargs[1:])

    nested_run_name = f"Model Metrics | engine={engine} | model={variant} "
    mean_accuracy = all_dataset_metrics["accuracy"].mean()
    mean_rmse_home = all_dataset_metrics["rmse_home"].mean()
    mean_mae_home = all_dataset_metrics["mae_home"].mean()
    mean_rmse_away = all_dataset_metrics["rmse_away"].mean()
    mean_mae_away = all_dataset_metrics["mae_away"].mean()
    mean_neg_log_likelihood = all_dataset_metrics["neg_log_likelihood"].mean()
    mean_brier_score = all_dataset_metrics["brier_score"].mean()
    mean_rps = all_dataset_metrics["rps"].mean()

    mean_model_metrics = {
        "accuracy": mean_accuracy,
        "rmse_home": mean_rmse_home,
        "mae_home": mean_mae_home,
        "rmse_away": mean_rmse_away,
        "mae_away": mean_mae_away,
        "neg_log_likelihood": mean_neg_log_likelihood,
        "brier_score": mean_brier_score,
        "rps": mean_rps,
    }
    return mean_model_metrics, nested_run_name


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
