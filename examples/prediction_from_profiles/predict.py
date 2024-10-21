from functools import partial
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

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


def predict(df: pd.DataFrame) -> list[pd.DataFrame]:
    ts_data = TimeSeriesData(
        df,
        encoder_features=model.encoder_features,
        decoder_features=model.decoder_features,
        aux_features=model.aux_features,
        target_id=model.target_id,
        history_horizon=len(df),  # HINT: All the data is input data for one predicition.
        forecast_horizon=model.forecast_horizon,  # HINT: while not recommended the model can be used to predict any number of steps, performance might behave unpredicable.
        preparation_steps=[  # HINT: Order is important
            partial(dh.set_to_hours, freq="1h"),  # HINT: adjust freq to forecast frequency
            partial(
                dh.fill_if_missing, periodicity=config.get("periodicity", 24)
            ),  # HINT: should be set in the config of the model
            partial(
                dh.extend_df, add_steps=model.forecast_horizon
            ),  # HINT: This extends the dataframe by the given amount of steps to be forecasted over
            dh.add_cyclical_features,
            dh.add_onehot_features,
            partial(dh.add_missing_features, all_columns=[*model.encoder_features, *model.decoder_features]),
            model.scalers.transform,
            dh.check_continuity,
        ],
    )
    ts_data.to_tensor()
    # data includes the targets aswell which will be all NaN and are discarded as they are no inputs to the model
    data_tensors = [tens.unsqueeze(0) for tens in ts_data[0]][
        :-1
    ]  # We need to "unsqueeze" these tensors as the model expects batched data.

    prediction = model.predict(*data_tensors)
    # Let's assume we are interested in the 90% interval of both predicted features.
    # 1. Turn the parametrized prediction into quantile predicitons
    quantiles = [0.95, 0.5, 0.05]
    q_pred = model.loss_metric.get_quantile_prediction(predictions=prediction, quantiles=quantiles).values

    # 2. For convenience put each feature in a dataframe
    result = []
    for n_feat, feat in enumerate(model.target_id):
        pred_df = pd.DataFrame(
            q_pred[0, :, n_feat, :]
            .detach()
            .numpy(),  # HINT: There is only one sample (first 0 index) and we only need the first subfeature (expected value)
            index=ts_data.data.index[-model.forecast_horizon :],  # HINT: We only need the future portion of the index
            columns=[f"q{quant}" for quant in quantiles],
        )
        if model.scalers is not None:
            # for col in pred_df.columns:
            pred_df = model.scalers.manual_inverse_transform(pred_df, scale_as=feat)
        result.append(pred_df)
    return result


def main():
    timeseries = pd.read_csv(data_path, sep=",", index_col="Time", parse_dates=True)  # HINT: adjust sep and index_col
    predictions = predict(timeseries)
    for prediction, feat in zip(predictions, model.target_id):
        prediction.to_csv(Path(__file__).parent / "example_prediciton.csv")
        # To plot we create values for each quantile. This is a method available in the loss used for training the model.
        prediction.plot(title=feat)
        plt.savefig(Path(__file__).parent / f"prediction_{feat}.png")


if __name__ == "__main__":
    main()
