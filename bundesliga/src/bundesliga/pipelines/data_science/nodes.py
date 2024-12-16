"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""
import pandas as pd

# from kedro_datasets_experimental.netcdf import NetCDFDataset
from modelbuilder import FootballModel

"""PYMC Model 1
    """


# nb_teams = len(teams)
# model1 = pm.Model()
# min_mu = 0.0001
# average_goals = 3.0


def model_1(model_input_table: pd.DataFrame) -> None:
    model = FootballModel(model_input_table)
    model.fit(
        X=model_input_table[["home_id", "away_id"]],
        goals=model_input_table[["home_goals", "away_goals"]],
    )
