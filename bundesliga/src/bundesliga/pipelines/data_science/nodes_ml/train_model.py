import typing as t

import arviz as az
import bundesliga.model.pymc.pymc_simple_model as pm_simple
import bundesliga.model.pymc.pymc_toto_model as pm_toto
import bundesliga.model.pyro.pyro_simple_model as pyro_simple
import pandas as pd
import xarray as xr
from utils import brier_score, log_likelihood, rmse_mae, rps


def init_model(team_lexicon, parameters: t.Dict, **kwargs) -> xr.Dataset:
    model = parameters["model"]
    engine = parameters["engine"]

    match engine:
        case "pymc":
            match model:
                case "simple":
                    return pm_simple.SimpleModel(parameters)
                case "toto":
                    try:
                        kwargs["toto"]
                    except KeyError:
                        raise ValueError(
                            "pm_toto.TotoModel requires 'toto' as keyword argument."
                        )
                    return pm_toto.TotoModel(toto=kwargs["toto"])
        case "pyro":
            match model:
                case "simple":
                    return pyro_simple.SimplePyroModel(team_lexicon, parameters)
                case "toto":
                    raise NotImplementedError("Pyro not implemented.")
        case "stan":
            match model:
                case "simple":
                    raise NotImplementedError("Stan not implemented.")
                case "toto":
                    raise NotImplementedError("Stan not implemented.")
        case _:
            raise ValueError(f"Model {model} not supported.")


def train(model, train_data, parameters: t.Dict, **kwargs) -> xr.Dataset:
    X = train_data[["home_id", "away_id"]]
    y = train_data[["home_goals", "away_goals", "toto"]]

    idata = model.train(X=X, y=y, parameters=parameters)

    return model, idata


def predict_goals(model, test_data, parameters, **kwargs):
    pp = model.predict_goals(test_data)

    return pp


def evaluate(model, day, test_data, predictions):
    test_goal = test_data[["home_goals", "away_goals"]]
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

    winner_accuracy = (
        test_goal["true_result"].values == predictions["predicted_result"].values
    ).mean()

    results = {
        "winner_accuracy": winner_accuracy,
        "rmse_home": rmse_home,
        "mae_home": mae_home,
        "rmse_away": rmse_away,
        "mae_away": mae_away,
        "neg_log_likelihood": negative_log_likelihood,
        "brier_score": brier_score_,
        "rps": rps_mean,
    }
    return results


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
