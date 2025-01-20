from copy import deepcopy
from pathlib import Path

import numpy as np
import omegaconf
import pandas as pd
import scipy.stats
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

min_mu = 0.0001
low = 10e-8  # Constant


def split_time_data(vectorized_data: pd.DataFrame, current_day):
    """
    Gibt train_X, train_y, test_X, test_y zurück,
    basierend auf der Spalte 'matchday' in X / y.
    """
    current_day = int(current_day)
    train_data = vectorized_data[:current_day]
    test_data = vectorized_data[current_day : current_day + 1]

    return train_data, test_data


def load_config():
    proj_path = Path(__file__).parent.parent
    conf_path = str(proj_path / settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path)

    return conf_loader


def merge_dicts(dict1, dict2):
    """
    Recursively merge two dictionaries.

    Args:
        dict1 (dict): The first dictionary to merge.
        dict2 (dict): The second dictionary to merge.

    Returns:
        dict: The merged dictionary.
    """
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if (
            key in result
            and isinstance(result[key], omegaconf.dictconfig.DictConfig)
            and isinstance(value, omegaconf.dictconfig.DictConfig)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def get_teams(model):
    return model.team_lexicon


def get_goal_distribution(diff, max_goals=20):
    poisson_goals = np.zeros(max_goals)
    k = np.arange(0, max_goals)
    for lambda_ in diff:
        lambda_ = max(low, lambda_)
        poisson_goals += scipy.stats.poisson.pmf(k, lambda_)
    poisson_goals = poisson_goals / poisson_goals.sum()
    return poisson_goals


def brier_score(test_goal, toto_probs):
    # -------------------------------------------------------------------------
    # 6) Brier Score (3 Klassen: home, away, tie)
    # -------------------------------------------------------------------------
    # One-Hot-Encodierung für den wahren Ausgang
    # y_true = [1,0,0] bei home, [0,1,0] bei away, [0,0,1] bei tie
    # y_pred = [p_home, p_away, p_tie]
    # Brier = mean( sum( (y_true - y_pred)^2 ) ) pro Spiel

    one_hot = pd.get_dummies(test_goal["true_result"])  # Gibt evtl. nur 1 Spalte zurück

    # 2) Auf die drei erwarteten Spalten reindexen, fehlende mit 0 füllen
    one_hot = one_hot.reindex(columns=["home", "away", "tie"], fill_value=0)

    # 3) Brier Score: Für jede Zeile (Spiel) Summation (y_true - y_pred)^2
    #    Anschließend den Mittelwert (hier nur 1 Spiel -> 1 Zeile).
    brier_per_game = ((one_hot - toto_probs[["home", "away", "tie"]]) ** 2).sum(axis=1)
    brier_score_value = brier_per_game.mean()
    return brier_score_value


def rps(test_goals, toto_probs):
    # Wende diese Funktion pro Zeile an
    rps_values = []
    for i in range(len(test_goals)):
        # Weg 1: direkt aus DataFrame
        actual = test_goals["true_result"].iloc[i]
        p_away = toto_probs["away"].iloc[i]
        p_tie = toto_probs["tie"].iloc[i]
        p_home = toto_probs["home"].iloc[i]
        rps_val = rps_3(p_away, p_tie, p_home, actual)
        rps_values.append(rps_val)

    rps_mean = np.mean(rps_values)
    return rps_mean


def rps_3(prob_away, prob_tie, prob_home, actual):
    # -------------------------------------------------------------------------
    # 7) RPS (Ranked Probability Score) für 3 Klassen
    # -------------------------------------------------------------------------
    # Die Kategorien sind ordinal: away (0), tie (1), home (2)
    # -> wir müssen uns auf *eine* Reihenfolge einigen. Beispiel: away < tie < home.
    # Dann berechnen wir kumulative Wahrscheinlichkeiten.
    # away:   F_0 = p_away
    # away+tie: F_1 = p_away + p_tie
    # am Ende: F_2 = 1
    #
    # Genauso für das "true"-Ergebnis:
    # Ist das Ergebnis "away"? => O_0 = 1, O_1 = 1, O_2 = 1
    # Ist es "tie"?  => O_0 = 0, O_1 = 1, O_2 = 1
    # Ist es "home"? => O_0 = 0, O_1 = 0, O_2 = 1
    #
    # RPS pro Spiel: sum_{k=0..(C-2)} (F_k - O_k)^2
    # Für 3 Kategorien also k=0..1. Dann Mittelwert über alle Spiele.

    """
    Berechnet den RPS für ein einzelnes Spiel mit 3 möglichen Ausgängen.
    'actual' ist in {away, tie, home}.
    """
    # 1) Kumulative predicted distribution in der Reihenfolge (away, tie, home)
    F0 = prob_away
    F1 = prob_away + prob_tie
    # F2 = 1.0  # theoretisch, brauchen wir nur bis (C-2) = 1

    # 2) Kumulative "wahre" Verteilung
    # away -> O0=1, O1=1
    # tie  -> O0=0, O1=1
    # home -> O0=0, O1=0
    if actual == "away":
        O0, O1 = 1.0, 1.0
    elif actual == "tie":
        O0, O1 = 0.0, 1.0
    else:  # home
        O0, O1 = 0.0, 0.0

    rps = (F0 - O0) ** 2 + (F1 - O1) ** 2
    return rps


def log_likelihood(toto_probs, test_goals):
    # -------------------------------------------------------------------------
    # 5) (Negative) Log-Likelihood
    # -------------------------------------------------------------------------
    # Hier nehmen wir an, dass der tatsächliche Ausgang "home"/"away"/"tie"
    # in test_goals["true_result"] steht und
    # die entsprechende Wahrscheinlichkeit in toto_probs["home"/"away"/"tie"].

    # Mapping: "home" -> p_home, "away" -> p_away, "tie" -> p_tie
    # Dann LL pro Spiel: log(prob_for_true_outcome)
    # Negative Log-Likelihood = - \sum log(prob_for_true_outcome)
    # Wir können das Ganze mitteln -> average NLL
    merged = pd.concat([toto_probs, test_goals["true_result"]], axis=1)
    # Kleiner Schutz gegen p=0
    EPS = 1e-12

    merged["prob_of_true_result"] = merged.apply(get_probability, axis=1)
    merged["log_prob"] = np.log(merged["prob_of_true_result"] + EPS)

    negative_log_likelihood = -merged["log_prob"].mean()  # gemittelte NLL
    return negative_log_likelihood


def get_probability(row):
    # row hat columns: home, away, tie, true_result
    return row[row["true_result"]]  # row["home"] / row["away"] / row["tie"]


def rmse_mae(predictions, test_data):
    # -------------------------------------------------------------------------
    # 4) RMSE / MAE der Torprognosen
    # -------------------------------------------------------------------------

    predictions = predictions.reset_index(drop=True)

    # Sicherheitshalber: 'test_data' könnte ebenfalls Index haben
    # -> Das Problem: "Can only compare identically-labeled Series objects"
    test_data = test_data.reset_index(drop=True)

    # a) Home

    errors_home = predictions["home_goals"].values - test_data["home_goals"].values[0]
    rmse_home = np.sqrt(np.mean((errors_home) ** 2))
    mae_home = np.mean(np.abs(errors_home))

    # b) Away
    errors_away = predictions["away_goals"].values - test_data["away_goals"].values[0]
    rmse_away = np.sqrt(np.mean((errors_away) ** 2))
    mae_away = np.mean(np.abs(errors_away))

    return rmse_home, mae_home, rmse_away, mae_away
