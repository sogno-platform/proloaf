import os
import logging
import logging.config
import functools
import time
from proloaf.confighandler import read_config

MAIN_PATH = os.path.dirname(os.path.abspath(__file__))


def create_event_logger(
        name: str,
        config_path: os.path = os.path.join(MAIN_PATH, 'event_logging_conf.json'),
        default_logging_level=logging.INFO
) -> logging.Logger:
    """
    Creates an event logger for a specific python file.
    The configuration for the logger should be made in the event logging config file in .json format.
    Ideally, new loggers for a file should be added under their file name in the config file
        and this function should then be called via create_event_logger(__name__).
    If there is no logger configured for a file, a default logger is created.
    To avoid double logs, only set handlers in the "root" section, or turn off propagation,
     by setting the propagate attribute to False.
    For further information how to create and adjust the logging configuration see:
    https://docs.python.org/3/howto/logging.html#loggers

    Parameters
    ----------
    default_logging_level: logging.DEBUG / logging.INFO / logging.WARNING / logging.ERROR / logging.CRITICAL
        determines the default logging level, if no specification can be found in the logging config
    config_path: Path
        The path to the .json config file for logging
    name: String
        the name of the logger in the config; ideally the __name__ of the file.

    Returns
    -------
    Object of the Logger class
    """
    try:
        event_log_conf = read_config(
            config_path=config_path
        )
        #if 'file' in event_log_conf['handlers']:
        logging.config.dictConfig(event_log_conf)

    except FileNotFoundError:
        logging.basicConfig(level=default_logging_level)
        logger = logging.getLogger()
        logger.warning('Configuration file could not be found at the given path. Default logger was created.')
    else:
        if name not in event_log_conf["loggers"]:
            logging.basicConfig(level=default_logging_level)
            logger = logging.getLogger()
            logger.warning(
                'Logger name "{:s}" could not be found in the logging config. Default logger was created'.format(name)
            )
        else:
            logger = logging.getLogger(name)

    return logger


def timer(logger: logging.Logger):
    """
    Decorator which passes the runtime of the decorated function to a logger.
    The logger then prints the name and runtime of the function, if the logger is specified to print statements
    of rank "INFO".
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            logger.info(f"Finished {func.__name__!r} in {run_time:.4f} secs")
            return value
        return wrapper_timer
    return decorator
