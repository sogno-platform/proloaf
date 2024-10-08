from functools import partial
from pathlib import Path
import datetime
import pandas as pd

from proloaf import modelhandler as mh
from proloaf.confighandler import read_config
from proloaf.tensorloader import TimeSeriesData
from proloaf import datahandler as dh

# Path to the pro
config_path = Path(__file__).parent / "config.json"
model_path = Path(__file__).parent / "opsd_recurrent.pkl"
data_path = Path(__file__).parent / "data_input.csv"
config = read_config(config_path=config_path)
# data = pd.read_csv(data_path, sep=";")

model = mh.ModelHandler.load_model(model_path)


def predict(df: pd.DataFrame):
    config["history_horizon"] = len(df)
    ts_data = TimeSeriesData(
        df,
        preparation_steps=[ # HINT: Order is important
            partial(dh.set_to_hours, freq="1h"),  # HINT: adjust freq to forecast frequency
            partial(
                dh.fill_if_missing, periodicity=config.get("periodicity", 24)
            ),  # HINT: should be set in the config of the model
            partial(
                dh.extend_df, add_steps=24
            ),  # HINT: This extends the dataframe by the given amount of steps to be forecasted over
            dh.add_cyclical_features,
            dh.add_onehot_features,
            partial(dh.add_missing_features, all_columns=[*config["encoder_features"], *config["decoder_features"]]),
            model.scalers.transform,
            dh.check_continuity,
        ],
        **config,
    )
    ts_data.to_tensor()
    # data includes the targets aswell which will be all NaN and are discarded as they are no inputs to the model
    data_tensors = [tens.unsqueeze(0) for tens in ts_data[0]][:-1]

    prediciton = model.predict(*data_tensors)

    pred_df = pd.DataFrame(
        prediciton[0, :, :, 0].detach().numpy(),
        index=ts_data.data.index[-24:],
        columns=config["target_id"],  # HINT: The last 24 timesteps are the forecast horizon extenden earlier
    )
    if model.scalers is not None:
        for col in pred_df.columns:
            pred_df[col] = model.scalers.manual_inverse_transform(pred_df[[col]], scale_as=col)
    return pred_df


def main():
    timeseries = pd.read_csv(data_path, sep=",", index_col="Time", parse_dates=True)  # HINT: adjust sep and index_col
    prediction = predict(timeseries)
    print(prediction)
    prediction.to_csv("example_prediciton.csv")


if __name__ == "__main__":
    main()
