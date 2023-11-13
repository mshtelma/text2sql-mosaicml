# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt
# MAGIC %pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python
# COMMAND ----------


import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from text2sql.utils import setup_logging, get_dbutils

setup_logging()

get_dbutils().widgets.text("dbfs_path_to_model", "", "dbfs_path_to_model")
# COMMAND ----------
dbfs_path_to_model = get_dbutils().widgets.get("dbfs_path_to_model")

# COMMAND ----------
config = AutoConfig.from_pretrained(dbfs_path_to_model, trust_remote_code=True)
config.attn_config[
    "attn_impl"
] = "triton"  # change this to use triton-based FlashAttention
config.init_device = "cuda:0"  # For fast initialization directly on GPU!

model = AutoModelForCausalLM.from_pretrained(
    dbfs_path_to_model,
    config=config,
    torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(dbfs_path_to_model)
# COMMAND ----------


q = """
You are a SQL generating assistant. Generate SQL queries that can answer questions defined in the instruction. 
### Instruction:
-- Schema
CREATE TABLE table_4053 (
    "Executed person" text,
    "Date of execution" text,
    "Place of execution" text,
    "Crime" text,
    "Method" text,
    "Under President" text
)
------
-- Query
Under which president was gunther volz executed?
------
### Response:
"""
# COMMAND ----------


with torch.autocast("cuda", dtype=torch.bfloat16):
    inputs = tokenizer(q, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
