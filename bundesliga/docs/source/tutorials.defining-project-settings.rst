.. _tutorials-defining-project-settings:

Defining Project Settings
=========================

This tutorial guides you through setting up the project-related configurations before running the pipeline.

1. **Include Datasets**: Place your raw football datasets (CSV files) in the follwowing directory. 

``01_raw/football-datasets``

2. **Specify Datasets**: List the datasets you want to process in the `data_sets_list.yml` file.

.. literalinclude:: ../../conf/base/datasets_list.yml
    :language: yaml

3. **Choose Engine and Model**: In `settings.py`, modify the `DYNAMIC_PIPELINES_MAPPING` dictionary to include or exclude desired engines and models.

.. literalinclude:: ../../src/bundesliga/settings.py
    :lines: 78-82
    :language: python


4. **Adjust global Parameters**: Fine-tune global parameters in the `parameters.yml` file as needed.


.. literalinclude:: ../../conf/base/parameters.yml
    :lines: 13-30
    :language: yaml


5. **Adjust Engine-specific Parameters**: Fine-tune engine parameters in the `parameters.yml` file as needed.

.. literalinclude:: ../../conf/base/parameters.yml
    :lines: 39-48
    :language: yaml

6. **Adjust Model-specific Parameters**: Fine-tune model parameters in the `parameters.yml` file as needed.

.. literalinclude:: ../../conf/base/parameters.yml
    :lines: 50-59
    :language: yaml

1. **Choose an Experiment name**: In `mlflow.yml`, modify the Experiment name variable to set the name of the experiment in MLflow.

.. literalinclude:: ../../conf/local/mlflow.yml
    :lines: 36-37
    :language: yaml