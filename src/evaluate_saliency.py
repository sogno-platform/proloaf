import os

import proloaf.event_logging as el
import proloaf.saliency_map as sm
from proloaf.cli import parse_basic

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


logger = el.create_event_logger("interpreter")


def main():
    args = parse_basic()
    saliency_map_handler = sm.SaliencyMapHandler(args.station)
    saliency_map_handler.create_saliency_map()
    saliency_map_handler.plot()
    saliency_map_handler.save()
    saliency_map_handler.plot_predictions()


if __name__ == "__main__":
    main()
