import functools
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)


def validate_dataframe(
    df_arg_name: str = "df_name",
    required_columns=None,
    required_index=None,
    allow_empty=False,
):
    """
    Decorator für Nodes, die mit Pandas DataFrames arbeiten.

    - Prüft, ob der erste Funktionsparameter ein DataFrame ist.
    - Falls required_columns angegeben sind, wird geprüft, ob diese Spalten existieren.
    - Falls allow_empty=False, wird geprüft, ob der DataFrame leer ist.
    - Misst die Laufzeit der Funktion (Performance-Tracking).

    Args:
        required_columns (list, optional): Liste von Spaltennamen, die existieren müssen.
        allow_empty (bool): Falls False, wird ein leerer DataFrame als Fehler behandelt.
    """
    required_columns = required_columns or []

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Wir versuchen zuerst, `df_arg_name` in kwargs zu finden
            if df_arg_name in kwargs:
                df = kwargs[df_arg_name]
            else:
                # Falls kein Keyword-Argument, schauen wir positional nach
                import inspect

                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                try:
                    index = params.index(df_arg_name)
                    df = args[index]
                except (ValueError, IndexError):
                    raise ValueError(
                        f"Could not find argument '{df_arg_name}' in {func.__name__}."
                    )

            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"DataFrame '{df_arg_name}' is not a DataFrame.")

            if not allow_empty and df.empty:
                raise ValueError(f"DataFrame '{df_arg_name}' is empty!")

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            if required_index and df.index.name != required_index:
                raise ValueError(
                    f"DataFrame '{df_arg_name}' must have index '{required_index}', but found '{df.index.name}'."
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator
