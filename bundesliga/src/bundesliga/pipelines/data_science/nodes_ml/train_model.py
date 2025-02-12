import pandas as pd
import xarray as xr

import bundesliga.model.pymc.pymc_simple_model as pm_simple
import bundesliga.model.pymc.pymc_toto_model as pm_toto
import bundesliga.model.pyro.pyro_simple_model as pyro_simple
from bundesliga.model.base_footballmodel import FootballModel
from bundesliga.utils.validation import validate_dataframe


@validate_dataframe(
    df_arg_name="team_lexicon",
    required_columns=["index"],
    required_index="team",
    allow_empty=False,
)
def init_model(
    team_lexicon: pd.DataFrame, model_options: dict, **kwargs
) -> FootballModel:
    """
    Initializes and returns a prediction model instance based on the specified engine and model type.

    This function selects and instantiates a model implementation using the provided team lexicon and model options.
    It supports different probabilistic programming engines ("pymc" or "pyro") and model variants (e.g., "simple" or "toto").
    For unsupported combinations or missing required keyword arguments (such as 'toto' for the "toto" model variant under pymc),
    the function raises an appropriate error.

    Args:
        team_lexicon (pd.DataFrame): A DataFrame mapping team names to unique indices, with team names as the index.
        model_options (dict): A dictionary containing model configuration options, including:
            - "model": The model variant to use (e.g., "simple", "toto").
            - "engine": The probabilistic programming engine ("pymc" or "pyro").
        **kwargs: Additional keyword arguments required for certain model variants.

    Returns:
        FootballModel: An instance of the initialized prediction model.

    Raises:
        ValueError: If required keyword arguments are missing or if the specified engine is not supported.
        NotImplementedError: If the requested model variant is not implemented for the specified engine.
    """

    model = model_options["model"]
    engine = model_options["engine"]

    match engine:
        case "pymc":
            match model:
                case "simple":
                    return pm_simple.SimplePymcModel(
                        model_options=model_options, team_lexicon=team_lexicon
                    )
                case "toto":
                    return pm_toto.TotoModel(
                        model_options=model_options,
                        team_lexicon=team_lexicon,
                        # toto=kwargs["toto"],
                    )
        case "pyro":
            match model:
                case "simple":
                    return pyro_simple.SimplePyroModel(
                        model_options=model_options, team_lexicon=team_lexicon
                    )
                case "simple2":
                    return pyro_simple.SimplePyroModel(
                        model_options=model_options, team_lexicon=team_lexicon
                    )
                case "toto":
                    raise NotImplementedError("Pyro-toto not implemented.")

        case _:
            raise ValueError(f"Engine {engine} not supported.")


@validate_dataframe(
    df_arg_name="train_data",
    required_columns=["home_id", "away_id", "home_goals", "away_goals", "toto"],
    allow_empty=False,
)
def train(
    model: FootballModel, train_data: pd.DataFrame, model_options: dict, **kwargs
) -> FootballModel:
    """
    Trains the provided prediction model using the training data.

    This function extracts the necessary input features and target values from the training DataFrame,
    then invokes the model's training method with the extracted data and model options. The trained model
    is then returned for further use.

    Args:
        model (FootballModel): The prediction model instance to be trained.
        train_data (pd.DataFrame): A DataFrame containing training data with columns:
            "home_id", "away_id", "home_goals", "away_goals", and "toto".
        model_options (dict): A dictionary of model configuration parameters to be used during training.
        **kwargs: Additional keyword arguments for model training.

    Returns:
        FootballModel: The trained model instance.
    """
    X = train_data[["home_id", "away_id"]]
    y = train_data[["home_goals", "away_goals", "toto"]]

    idata = model.train(X=X, y=y, parameters=model_options)

    return model


@validate_dataframe(
    df_arg_name="test_data",
    required_columns=["home_id", "away_id"],
    allow_empty=False,
)
def predict_goals(
    model: FootballModel, test_data: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """
    Predicts goals for home and away teams using the trained model.

    This function uses the provided test data, which should contain the columns "home_id" and "away_id",
    to generate predictions for the number of goals scored by each team. It returns the model's predictions.

    Args:
        model: The trained prediction model instance.
        test_data (pd.DataFrame): A DataFrame containing test data with at least "home_id" and "away_id" columns.
        **kwargs: Additional keyword arguments for the prediction process.

    Returns:
        The predictions generated by the model, typically in the form of an xarray Dataset or another structured format.
    """
    predictions = model.predict_goals(test_data)

    return predictions
