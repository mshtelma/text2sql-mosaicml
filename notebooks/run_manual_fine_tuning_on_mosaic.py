# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# COMMAND ----------
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------
import mcli
from mcli import finetune, RunConfig
from text2sql.utils import setup_logging, get_dbutils

setup_logging()

get_dbutils().widgets.text(
    "yaml_config", "../yaml/mosai-text2sql-mpt7b-nsql-fine-tuning.yaml", "yaml_config"
)

# COMMAND ----------

yaml_config = get_dbutils().widgets.get("yaml_config")

# COMMAND ----------

mcli.initialize(api_key=get_dbutils().secrets.get(scope="msh", key="mosaic-token"))

# COMMAND ----------
# MAGIC !mcli run -f  {yaml_config} --follow
