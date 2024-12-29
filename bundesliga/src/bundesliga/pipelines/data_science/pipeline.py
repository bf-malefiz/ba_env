"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import fit


def create_pipeline(**kwargs) -> Pipeline:
    base_data_science = pipeline(
        [
            node(
                func=fit,
                inputs=["x_data", "y_data", "params:model_parameters"],
                outputs="fit_idata",
                # outputs=["fit_idata", "fit_samples"],
                name="fit_node",
            ),
        ],
        # tags="tag1",  # Optional, each pipeline node will be tagged
        # namespace="",  # Optional
        # inputs={},  # Optional
        # outputs={},  # Optional
        # parameters={"params:model_parameters"},  # Optional
    )

    ds_pipeline_1 = pipeline(
        [
            base_data_science,
        ],
        inputs=["x_data", "y_data"],
        namespace="active_modelling_pipeline",
        parameters={"params:model_parameters": "params:active_model_parameters"},
    )

    # ds_pipeline_2 = pipeline(
    #     pipe=pipeline_instance,
    #     inputs="model_input_table",
    #     namespace="candidate_modelling_pipeline",
    # )

    # return ds_pipeline_1 + ds_pipeline_2
    return ds_pipeline_1
