"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=test,
                inputs=None,
                outputs="test_netcdf",
                name="test_node",
            ),
        ]
    )
