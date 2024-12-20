"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_1


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline(
        [
            node(
                func=model_1,
                inputs=["model_input_table", "params:model_parameters"],
                outputs="model_1_idata",
                name="model_1_node",
            ),
        ]
    )
    ds_pipeline_1 = pipeline(
        pipe=pipeline_instance,
        inputs="model_input_table",
        namespace="active_modelling_pipeline",
    )

    ds_pipeline_2 = pipeline(
        pipe=pipeline_instance,
        inputs="model_input_table",
        namespace="candidate_modelling_pipeline",
    )

    return ds_pipeline_1 + ds_pipeline_2
