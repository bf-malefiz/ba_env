"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_1


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_1,
                inputs="model_input_table",
                outputs=None,
                name="model_1_node",
            ),
        ]
    )
