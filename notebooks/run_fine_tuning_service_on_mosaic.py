# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC

# COMMAND ----------

import os.path
import time
import pathlib
import mcli
from mcli import RunStatus

from text2sql.utils import setup_logging, get_dbutils

setup_logging()

SUPPORTED_INPUT_MODELS = [
    "mosaicml/mpt-30b",
    "mosaicml/mpt-7b-8k",
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
    "base_model", "mosaicml/mpt-7b-8k", SUPPORTED_INPUT_MODELS, "base_model"
)
get_dbutils().widgets.text(
    "s3_data_path", "s3://msh-tmp1/t2s/nsql_dolly_prompt_v1_limit5000/", "s3_data_path"
)
get_dbutils().widgets.text(
    "s3_model_output_folder",
    "s3://msh-tmp1/t2s/models/mpt-7b-nsql-5k-ft-v1",
    "s3_model_output_folder",
)
get_dbutils().widgets.text("training_duration", "10ba", "training_duration")
get_dbutils().widgets.text("learning_rate", "1e-6", "learning_rate")

# COMMAND ----------

base_model = get_dbutils().widgets.get("base_model")
s3_data_path = get_dbutils().widgets.get("s3_data_path")
s3_model_output_folder = get_dbutils().widgets.get("s3_model_output_folder")
training_duration = get_dbutils().widgets.get("training_duration")
learning_rate = get_dbutils().widgets.get("learning_rate")

# COMMAND ----------

mcli.initialize(api_key=get_dbutils().secrets.get(scope="msh", key="mosaic-token"))


# COMMAND ----------

mcli.initialize(api_key=get_dbutils().secrets.get(scope="msh", key="mosaic-token"))
yaml_config = str(
    (pathlib.Path.cwd().parent / "yamls" / "mosaic-fine-tuning-service.yaml").absolute()
)
_run = mcli.create_finetuning_run(
    model=base_model,
    train_data_path=os.path.join(s3_data_path, "train.jsonl"),
    eval_data_path=os.path.join(s3_data_path, "val.jsonl"),
    save_folder=s3_model_output_folder,
    training_duration=training_duration,
    learning_rate=learning_rate,
    task_type="INSTRUCTION_FINETUNE",
    experiment_tracker={
        "mlflow": {
            "experiment_path": "/Shared/msh_test_exp",
            "model_registry_path": "msh.t2s.t2s_model",
        }
    },
)
print(f"Started Run {_run.name}. The run is in status {_run.status}.")

# COMMAND ----------

mcli.wait_for_run_status(_run.name, RunStatus.RUNNING)
for s in mcli.follow_run_logs(_run.name):
    print(s)

# COMMAND ----------


