import functools

import pandas as pd


def validate_dataframe(
    df_arg_name: str = "df_name",
    required_columns=None,
    required_index=None,
    allow_empty=False,
):
    """
    Decorator for nodes that work with Pandas DataFrames.

    This decorator validates that the argument specified by `df_arg_name` is a Pandas DataFrame.
    It checks that the DataFrame is non-empty (if allow_empty is False), contains the required columns,
    and has the expected index name (if required_index is provided).

    Args:
        df_arg_name (str, optional): The name of the DataFrame argument to validate. Defaults to "df_name".
        required_columns (list, optional): A list of column names that must be present in the DataFrame.
        required_index (str, optional): The expected name of the DataFrame's index.
        allow_empty (bool, optional): If False, the DataFrame must not be empty. Defaults to False.

    Returns:
        function: The decorated function with added DataFrame validation.
    """
    required_columns = required_columns or []

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if `df_arg_name` in kwargs
            if df_arg_name in kwargs:
                df = kwargs[df_arg_name]
            else:
                # else check positional arguments
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
                raise ValueError(
                    f"DataFrame '{df_arg_name}' is empty in {func.__name__}."
                )

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
