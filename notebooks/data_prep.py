# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# COMMAND ----------
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------
from text2sql.utils import setup_logging, get_dbutils
from text2sql.dataprep.nsql import (
    prepare_and_write_mds_for_nsql_dataset,
    prepare_and_write_jsonl_for_nsql_dataset,
)

setup_logging()

get_dbutils().widgets.combobox("output_type", "jsonl", ["jsonl", "MDS"], "output_type")
get_dbutils().widgets.text("limit", "20000", "limit")
get_dbutils().widgets.text(
    "dbfs_output_location",
    "/dbfs/mnt/mshtmp/t2s/nsql_dolly_prompt_v2",
    "dbfs_output_location",
)

# COMMAND ----------
output_type = get_dbutils().widgets.get("output_type")
limit = int(get_dbutils().widgets.get("limit"))
dbfs_output_location = get_dbutils().widgets.get("dbfs_output_location")

# COMMAND ----------
if output_type == "MDS":
    prepare_and_write_mds_for_nsql_dataset(dbfs_output_location)
else:
    prepare_and_write_jsonl_for_nsql_dataset(dbfs_output_location)

# COMMAND ----------