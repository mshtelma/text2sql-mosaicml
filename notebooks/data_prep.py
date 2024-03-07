# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

# COMMAND ----------

from text2sql.utils import setup_logging, get_dbutils
from text2sql.dataprep.nsql import (
    prepare_and_write_mds_for_nsql_dataset,
    prepare_and_write_jsonl_for_nsql_dataset,
)

setup_logging()

get_dbutils().widgets.combobox("output_type", "jsonl", ["jsonl", "MDS"], "output_type")
get_dbutils().widgets.text("limit", "5000", "limit")
get_dbutils().widgets.text(
    "dbfs_output_location",
    "dbfs:/Volumes/msh/t2s/t2s/data/nsql_dolly_prompt_v1",
    "dbfs_output_location",
)

# COMMAND ----------

output_type = get_dbutils().widgets.get("output_type")
limit = int(get_dbutils().widgets.get("limit"))
dbfs_output_location = get_dbutils().widgets.get("dbfs_output_location")

# COMMAND ----------

if limit > 0:
    dbfs_output_location = f"{dbfs_output_location}_limit{limit}"
if output_type == "MDS":
    prepare_and_write_mds_for_nsql_dataset(dbfs_output_location, limit=limit)
else:
    prepare_and_write_jsonl_for_nsql_dataset(dbfs_output_location, limit=limit)
