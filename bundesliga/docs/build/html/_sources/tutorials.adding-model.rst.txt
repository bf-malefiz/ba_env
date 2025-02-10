Adding Models to the Pipeline
=============================
The framework provides a modular design that allows you to easily add new models to the pipeline. The steps below outline how to add a new model to the framework and integrate it into the existing pipeline. 

1. **Create a new model class**: 

Create a new class for your model in the `bundesliga.model` module. The class should inherit from the :py:class:`~bundesliga.model.base_footballmodel.FootballModel` class and implement the required methods for model training and evaluation. For this example we will use the already implemented :py:class:`~bundesliga.model.pyro.pyro_simple_model.SimplePyroModel` class.

2. **Define the model parameters**:

You have to add the model to the `parameters.yml` file. Even if you don't change the parameters you need to add at least the namespace for it. The parameters are mapped to its name and used to initialize the model and to set the parameters for the model parameters. See the following code snippet showing a new model `simple2` added to the `parameters.yml` file.

.. literalinclude:: ../../conf/base/parameters.yml
    :lines: 102-107
    :language: yaml

See :ref:`Defining Project Settings <tutorials-defining-project-settings>` for more information on how to adjust the parameters.

3. **Add the model to the pipeline**:

For the model to be instantiated you need to add a match statement to the node: `train_model.py` for the engine pyro. The following code snippet shows how to add the `simple2` model to the pipeline. The key to match is the model name you defined in the `parameters.yml` file. If you developed a new model, you need to instantiate it here.

.. literalinclude:: ../../src/bundesliga/pipelines/data_science/nodes_ml/train_model.py
    :lines: 66-69

4. **Register the model in settings.py**:

Finaly you need to update the `DYNAMIC_PIPELINES_MAPPING` dictionary in the `settings.py` file. Add the new model to the dictionary with the corresponding engine and model class. The following code snippet shows how to add the `simple2` model to the pipeline. As the model is using the `pyro` engine, only the model has to be added to the dictionary values.

.. literalinclude:: ../../src/bundesliga/settings.py
    :lines: 78-82
    :language: python



Now the model is integrated and will be used in the `kedro run -pipeline "pyro"` command as well as in the default pipeline.