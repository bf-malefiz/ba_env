Running a Kedro Pipeline
=========================

**Running all engines at once**

Once you have defined your project settings, you can run the complete pipeline by using the default run method from Kedro:

.. code-block:: bash
    
    > kedro run

This will start the pipeline with all engines aggregated. It corresponds to the pipeline saved as "`__default__`" key in the `pipeline_registry.py` output dictionary. For more information on the registry see the :mod:`bundesliga.pipeline_registry` module.



**Running specific engine**


To run a specific engine, you can use the following command:

.. code-block:: bash
    
    > kedro run --pipeline <engine_name>

This will start the pipeline with the engine you choose. It corresponds to the pipeline saved as "`<engine_name>`" key in the :mod:`bundesliga.pipeline_registry.register_pipelines` output dictionary and the key in the `DYNAMIC_PIPELINES_MAPPING` dictionary in the `settings.py` file.


