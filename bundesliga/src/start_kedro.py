from pathlib import Path

import mlflow
from bundesliga import settings
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro_mlflow.mlflow.kedro_pipeline_model import KedroPipelineModel

bootstrap_project(Path.cwd())


with KedroSession.create() as session:
    session.run(pipeline_name="etl")
    session.close()
    mlflow.end_run()
with KedroSession.create() as session:
    session.run(pipeline_name="pymc")
    session.close()
    mlflow.end_run()
with KedroSession.create() as session:
    session.run(pipeline_name="pyro")
    session.close()
    mlflow.end_run()
