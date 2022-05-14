import os
import sys

sys.path.append("../")
MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)

from proloaf.saliency_map import SaliencyMapHandler
from proloaf.cli import parse_basic
from proloaf.event_logging import create_event_logger

logger = create_event_logger('interpreter')


def main():

    args = parse_basic()
    saliency_map_handler = SaliencyMapHandler(args.station)
    saliency_map_handler.create_saliency_map()
    saliency_map_handler.plot()
    saliency_map_handler.save()


if __name__ == "__main__":
    main()
