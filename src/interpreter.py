import os
import sys

import pandas as pd

sys.path.append("../")
MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_PATH)

from proloaf. explanation_methods import SaliencyMapUtil


def main():
    date = pd.to_datetime('27.07.2019 00:00:00', format="%d.%m.%Y %H:%M:%S")
    saliency_map = SaliencyMapUtil('opsd', date)
    saliency_map.optimize()
    saliency_map.plot()


if __name__ == "__main__":
    main()
