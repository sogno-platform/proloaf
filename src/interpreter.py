import os
import sys

sys.path.append("../")
MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)

from proloaf.explanation_methods import SaliencyMapUtil
from proloaf.cli import parse_basic


def main():

    args = parse_basic()
    saliency_map_util = SaliencyMapUtil(args.station)
    saliency_map_util.optimize()
    saliency_map_util.plot()
    saliency_map_util.save()


if __name__ == "__main__":
    main()
