import logging
import logging.config
from pathlib import Path

def setup_logging(default_path='logger.cfg.yml', default_level=logging.INFO):
    """
    Setup logging configuration from YAML file.
    Args:
        default_path (str): Path to the YAML configuration file.
        default_level (int): Default logging level if the configuration file is not found.
    """
    path = Path(__file__).parent / default_path
    if path.is_file():
        try:
            from ruamel.yaml import YAML
        except ImportError:
            logging.basicConfig(level=default_level)
            logging.warning("ruamel.yaml is not installed; using default logging.")
            return
        with open(path, 'rt') as f:
            config = YAML(typ="safe", pure=True).load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f"Warning: Logging configuration file '{default_path}' not found. Using default configurations.")

def get_logger(_name=None):
    """
    Return a configured logger.
    Args:
        _name (str): The module name.
    Returns:
        The logger instance.
    """
    return logging.getLogger(_name)