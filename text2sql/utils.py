import logging

from pyspark.sql import SparkSession
from gql.transport.requests import log as requests_logger
from gql.transport.websockets import log as websockets_logger

logger = logging.getLogger(__name__)


def get_spark() -> SparkSession:
    try:
        import IPython

        return IPython.get_ipython().user_ns["spark"]
    except:
        raise Exception(
            "Spark is not available! You are probably running this code outside of Databricks environment."
        )


def get_dbutils():
    """
    Returns dbutils object. Works only on Databricks.
    """
    try:
        import IPython

        return IPython.get_ipython().user_ns["dbutils"]
    except:
        raise Exception(
            f"Could not retrieve dbutils because not running in a Databricks notebook!"
        )


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("sh.command").setLevel(logging.ERROR)

    requests_logger.setLevel(logging.WARNING)
    websockets_logger.setLevel(logging.WARNING)
