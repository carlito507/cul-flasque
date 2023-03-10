from pathlib import Path
import logging
import sys

from pprint import pformat
import yaml
import os

yaml_dir = Path(__file__).resolve().parent
yaml_path = yaml_dir / "config.yaml"

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def load_yaml_config(path):
    try:
        with open(path, "r") as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        logging.error(f"failed to load {path}: {exc}")
        return None


yaml_config = load_yaml_config(yaml_path)
if yaml_config is not None:
    logging.info(f"loaded config from {yaml_path}:")
    logging.info(pformat(yaml_config))
    globals().update(yaml_config)
else:
    logging.error(f"could not load config from {yaml_path}.")
    sys.exit(1)

SERVER_PORT = yaml_config.get("SERVER_PORT", None)
SERVER_DIR = Path(__file__).resolve().parent


