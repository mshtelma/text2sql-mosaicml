# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# COMMAND ----------
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
import os.path

# COMMAND ----------
import pathlib
import mcli
from text2sql.utils import setup_logging, get_dbutils

setup_logging()

SUPPORTED_INPUT_MODELS = [
    "mosaicml/mpt-30b-instruct",
    "mosaicml/mpt-7b-8k-instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-13b-hf",
    "codellama/CodeLlama-34b-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
]
get_dbutils().widgets.combobox(
    "base_model", "mosaicml/mpt-7b-8k-instruct", SUPPORTED_INPUT_MODELS, "base_model"
)
get_dbutils().widgets.text(
    "s3_data_path", "s3://msh-tmp1/t2s/nsql_dolly_prompt_v2/", "s3_data_path"
)
get_dbutils().widgets.text(
    "s3_model_output_folder",
    "s3://msh-tmp1/t2s/models/mpt-1b-nsql-20k-v1-saas",
    "s3_model_output_folder",
)
get_dbutils().widgets.text("training_duration", "4ba", "training_duration")
get_dbutils().widgets.text("learning_rate", "1e-6", "learning_rate")

# COMMAND ----------

base_model = get_dbutils().widgets.get("base_model")
s3_data_path = get_dbutils().widgets.get("s3_data_path")
s3_model_output_folder = get_dbutils().widgets.get("s3_model_output_folder")
training_duration = get_dbutils().widgets.get("training_duration")
learning_rate = get_dbutils().widgets.get("learning_rate")

# COMMAND ----------

mcli.initialize(api_key=get_dbutils().secrets.get(scope="msh", key="mosaic-token"))
yaml_config = str(
    (pathlib.Path.cwd().parent / "yamls" / "mosaic-fine-tuning-service.yaml").absolute()
)

# COMMAND ----------

# MAGIC !mcli finetune -f {yaml_config} \
# MAGIC --model {base_model} \
# MAGIC --train-data-path {os.path.join(s3_data_path, "train.jsonl")} \
# MAGIC --eval-data-path {os.path.join(s3_data_path, "val.jsonl")} \
# MAGIC --save-folder {s3_model_output_folder} \
# MAGIC --training-duration {training_duration} \
# MAGIC --learning-rate {learning_rate} \
# MAGIC --context-length 8192 \
# MAGIC --follow
