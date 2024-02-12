# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import mcli
from mcli import RunConfig, RunStatus
from text2sql.utils import setup_logging, get_dbutils

setup_logging()

get_dbutils().widgets.text(
    "yaml_config", "../yamls/mosaic-text2sql-mpt7b-nsql-fine-tuning.yaml", "yaml_config"
)

# COMMAND ----------

mcli.initialize(api_key=get_dbutils().secrets.get(scope="msh", key="mosaic-token"))
yaml_config = get_dbutils().widgets.get("yaml_config")
_run = mcli.create_run(RunConfig.from_file(yaml_config))
print(f"Started Run {_run.name}. The run is in status {_run.status}.")

# COMMAND ----------

mcli.wait_for_run_status(_run.name, RunStatus.RUNNING)
for s in mcli.follow_run_logs(_run.name):
    print(s)

# COMMAND ----------


