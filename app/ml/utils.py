import logging.config

from app.ml.constants import Constants as C


def configure_main_logger(logger_name):
    logging.config.fileConfig(
        C.LOGGING_PATH / C.LOGGING_CONFIG_FILENAME, disable_existing_loggers=False
    )
    logger = logging.getLogger(logger_name)
    return logger


def flatten_dict_to_string(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict_to_string(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_to_string(d, sep="_"):
    flat_dict = flatten_dict_to_string(d, sep=sep)
    return sep.join(f"{k}{sep}{v}" for k, v in flat_dict.items())


def create_label(method_name, config_dict, sep="_"):
    config_string = dict_to_string(config_dict, sep=sep)
    config_string = config_string.replace(".", "p")
    return f"{method_name}{sep}{config_string}"
