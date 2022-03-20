from pathlib import Path
import os
from copy import deepcopy
from proloaf.cli import parse_basic
from proloaf.confighandler import read_config, write_config
from proloaf.event_logging import create_event_logger

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ARGS = parse_basic()

logger = create_event_logger(__name__)


def patch_main_0_1_to_0_2(main_path, config_path):
    config = read_config(config_path=config_path, main_path=main_path)
    logger.info(f"patching: {config_path}")
    new_config = deepcopy(config)
    # move model specfic configuration into modelparameters and add selection
    recurrent_model_keys = [
        "core_net",
        "core_layers",
        "dropout_fc",
        "dropout_core",
        "rel_linear_hidden_size",
        "rel_core_hidden_size",
        "relu_leak",
    ]
    recurrent_model_config = {
        k: new_config.pop(k, None) for k in recurrent_model_keys if new_config.get(k)
    }
    if "model_parameters" not in new_config:
        new_config["model_parameters"] = {"recurrent": recurrent_model_config}
    else:
        logger.info(
            "The input config was aleady (partially) converted before,"
            + " the patcher tried its best to get a working config, definetly check it afterwards by hand."
        )
        if "recurrent" not in new_config["model_parameters"]:
            new_config["model_parameters"]["recurrent"] = recurrent_model_config
        else:
            new_config["model_parameters"]["recurrent"].update(recurrent_model_config)
    if not new_config.get("model_class"):
        new_config["model_class"] = "recurrent"
    related_config = new_config.get("exploration_path")
    if related_config:
        patch_main_0_1_to_0_2(main_path, related_config)
    # Remove deprecated score keeping
    new_config.pop("best_loss", None)
    new_config.pop("best_score", None)
    old_path = os.path.join(main_path, config_path)
    new_path = old_path.rsplit(".", 1)
    new_path = f"{new_path[0]}_old.{new_path[1]}"
    os.rename(old_path, new_path)
    write_config(new_config, config_path=config_path, main_path=main_path)


if __name__ == "__main__":
    if ARGS.config:
        if ARGS.config.endswith(".json"):
            patch_main_0_1_to_0_2(main_path=MAIN_PATH, config_path=ARGS.config)
        else:
            p = Path(os.path.join(MAIN_PATH, ARGS.config)).rglob("*config.json")
            for conf in p:
                patch_main_0_1_to_0_2(main_path=MAIN_PATH, config_path=conf)
    else:
        patch_main_0_1_to_0_2(
            main_path=MAIN_PATH,
            config_path=os.path.join("targets", ARGS.station, "config.json"),
        )
