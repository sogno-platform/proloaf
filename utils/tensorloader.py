
import numpy as np
import torch
import sklearn

import plf_util.datatuner as dt

class CustomTensorData:
    def __init__(self, inputs1, inputs2, targets):
        self.inputs1 = torch.Tensor(inputs1).float()
        self.inputs2 = torch.Tensor(inputs2).float()
        self.targets = torch.Tensor(targets).float()

    def __getitem__(self, index):
        return self.inputs1[index, :, :], self.inputs2[index, :, :], self.targets[index, :, :]

    def __len__(self):
        return self.targets.shape[0]

    def to(self, device):
        self.inputs1 = self.inputs1.to(device)
        self.inputs2 = self.inputs2.to(device)
        self.targets = self.targets.to(device)
        return self


class CustomTensorDataLoader:
    def __init__(self, dataset: CustomTensorData, batch_size=1, shuffle=False, drop_last=True):
        if not drop_last:
            raise NotImplementedError

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        n = self.batch_size * (len(self.dataset) // self.batch_size)
        if self.shuffle:
            permutation = np.random.permutation(len(self.dataset))[:n]
        else:
            permutation = np.arange(n)
        self.permutation = np.reshape(permutation, (len(self), self.batch_size))
        self.batch_index = 0
        return self

    def __next__(self):
        if self.batch_index < len(self):
            indices = self.permutation[self.batch_index]
            self.batch_index += 1
            return self.dataset[indices]
        else:
            raise StopIteration

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def to(self, device: str):
        self.dataset.to(device)
        return self

    def number_features1(self):
        return self.dataset.inputs1.shape[2]

    def number_features2(self):
        return self.dataset.inputs2.shape[2]


def make_dataloader(df, target_id, encoder_features, decoder_features, history_horizon,
                    forecast_horizon, batch_size = 1, shuffle = True, drop_last = True, **_):

    x_enc = dt.extract(df[encoder_features].iloc[:-forecast_horizon, :], history_horizon)
    # shape input data that is known for the Future, here take perfect hourly temp-forecast
    x_dec = dt.extract(df[decoder_features].iloc[history_horizon:, :], forecast_horizon)
    # shape y
    y = dt.extract(df[[target_id]].iloc[history_horizon:,:], forecast_horizon)

    custom_tensor_data = CustomTensorData(x_enc, x_dec, y)
    return CustomTensorDataLoader(custom_tensor_data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
