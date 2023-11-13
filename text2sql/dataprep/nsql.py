import os
from datasets import load_dataset
from pyspark.sql import SparkSession, DataFrame
from streaming.base.converters import dataframe_to_mds

from text2sql.utils import get_spark


def load_nsql_dataset(spark: SparkSession, limit: int = -1) -> DataFrame:
    pdf = load_dataset("NumbersStation/NSText2SQL")["train"].to_pandas()
    if limit > 0:
        pdf = pdf[:limit]
    sdf = spark.createDataFrame(pdf)
    return sdf


def format_prompt(schema: str, query: str) -> str:
    INSTRUCTION_KEY = "### Instruction:"
    RESPONSE_KEY = "### Response:"
    INTRO_BLURB = "You are a SQL generating assistant. Generate SQL queries that can answer questions defined in the instruction. "
    PROMPT_FOR_GENERATION_FORMAT = """{intro}
        {instruction_key}
        -- Schema
        {schema}
        ------
        -- Query
        {query}
        ------
        {response_key}
        """
    return PROMPT_FOR_GENERATION_FORMAT.format(
        intro=INTRO_BLURB,
        instruction_key=INSTRUCTION_KEY,
        schema=schema,
        query=query,
        response_key=RESPONSE_KEY,
    )


def transform_nsql_row(row) -> str:
    arr = row.instruction.split("--")
    return format_prompt(arr[0].strip(), arr[2].strip())


def transform_nsql_udf(iterator):
    for df in iterator:
        df["prompt"] = df.apply(transform_nsql_row, axis=1)
        df["response"] = df.output
        df = df[["prompt", "response"]]
        yield df


def prepare_nsql_dataset(spark: SparkSession, limit: int) -> DataFrame:
    sdf = load_nsql_dataset(spark, limit)
    transformed_sdf = sdf.mapInPandas(
        transform_nsql_udf, schema="prompt string, response string"
    )
    return transformed_sdf


def store_as_mds(sdf: DataFrame, path: str):
    dataframe_to_mds(
        sdf.repartition(8),
        merge_index=True,
        mds_kwargs={"out": path, "columns": {"prompt": "str", "response": "str"}},
    )


def prepare_and_write_mds_for_nsql_dataset(
    output_path: str,
    limit: int = -1,
):
    spark = get_spark()
    transformed_sdf = prepare_nsql_dataset(spark, limit)
    train_sdf, val_sdf = transformed_sdf.randomSplit([0.9, 0.1])
    store_as_mds(train_sdf, os.path.join(output_path, "train"))
    store_as_mds(val_sdf, os.path.join(output_path, "val"))


def prepare_and_write_jsonl_for_nsql_dataset(
    output_path: str,
    limit: int = -1,
):
    spark = get_spark()
    transformed_sdf = prepare_nsql_dataset(spark, limit)
    train_sdf, val_sdf = transformed_sdf.randomSplit([0.9, 0.1])
    train_sdf.toPandas().to_json(
        os.path.join(output_path, "train.jsonl"), orient="records", lines=True
    )

    val_sdf.toPandas().to_json(
        os.path.join(output_path, "val.jsonl"), orient="records", lines=True
    )
