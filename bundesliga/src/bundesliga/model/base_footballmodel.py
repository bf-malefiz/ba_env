"""
Module: football_model.py

Summary:
    An abstract base class for football match prediction models.

Dependencies:
    - abc (ABC, abstractmethod)
    - typing (Annotated, Any, Dict, Union)
    - numpy as np
    - pandas as pd
"""

from abc import ABC, abstractmethod
from typing import Annotated, Any, Union

import numpy as np
import pandas as pd


class FootballModel(ABC):
    """
    An abstract base class for football match prediction models.

    This class defines the interface for training and predicting match outcomes.
    Subclasses must implement the `train` and `predict_toto_probabilities` methods.

    Methods:
        train(X, y, **kwargs): Abstract train method using the provided data.
        predict_toto_probabilities(predictions, **kwargs): Predicts match outcome probabilities.
        _validate_output(output): Validates the output of predictions.
    """

    @abstractmethod
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
        **kwargs,
    ) -> None:
        """
        Ensures the implementation of a training method for the model.

        Args:
            X: Needed input features for training. The exact format depends on the implementation.
            y: Needed target labels for training. The exact format depends on the implementation.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        pass

    @abstractmethod
    def predict_toto_probabilities(
        self, predictions: Union[pd.DataFrame, np.ndarray, dict[str, Any]], **kwargs
    ) -> Annotated[np.ndarray, "shape=(3,)", "dtype=float"]:
        """
        Ensures the implementation to predict the probabilities of match outcomes (toto).

        Args:
            predictions: Needed input data for making predictions. The exact format depends on the implementation.
            **kwargs: Additional keyword arguments for prediction.

        Returns:
            np.ndarray: An array of shape (3,) containing the probabilities for:
                - Team 1 winning (index 0),
                - Team 2 winning (index 1),
                - A draw (index 2).

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def _validate_output(
        self, output: np.ndarray, expected_shape: tuple = (1, 3)
    ) -> None:
        """
        Validates the output of predictions at runtime.

        Args:
            output (np.ndarray): The output array to validate.
            expected_shape (tuple): The expected shape of the output. Default is (1, 3).

        Raises:
            TypeError: If the output is not a NumPy array or does not contain floats.
            ValueError: If the output does not have the correct shape.
        """

        if not isinstance(output, np.ndarray):
            raise TypeError("Output must be a NumPy array")
        if output.shape[-len(expected_shape) :] != expected_shape:
            raise ValueError(
                f"Output must have shape {expected_shape}, but got {output.shape}"
            )
        if output.dtype != np.float64:
            raise TypeError("Output must contain floats")
