#### Timeseries Settings

ProLoaF is a machine-learning based timeseries forecasting project. The supervised learning requires data with 
a (pandas) datetime index. Typical time resolutions are: `ms`, `s`, `m`, `h`, `d`.
Endogenous (lagged) inputs and exogenous (=explanatory) variables that affect the future explained variable, 
are split into multiple windows with the sequence length of ``history_horizon`` and fed to the encoder.
For better understanding, we recommend 
[the illustrative guide](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb) 
on a similar sequence-to-sequence architecture authored by Ben Trevett.

The time step size is equal to the underlying timeseries data resolution. 
The example files apply day-ahead forecasting in hourly resolution. 
> **_Note:_** A forecast that produces a predicted sequence starting from the forecasting execution time and including the next day, 
is >=24h, depending on the forecast execution time t, e.g. a day-ahead forecast executed at 9 am, shall produce a 40 hour horizon.

#### Tensorboard utilization

This project uses Tensorboard to display in-depth information about each train run. Information about the run itself like training time and validation
loss are visible in the `Scalars` tab. The `HParams` tab allows sorting of runs by hyper parameters as well as parallel and scatter plots. 

Install Tensorboard and run the Tensorboard command-line interface as described [here](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html). The default directory that runs get saved in is the `runs/` directory.
To display runs in Tensorboard, use `tensorboard --logdir=runs`, then open a browser and navigate to [127.0.0.1:6006](127.0.0.1:6006)


#### Parameter List

**Shell Params Upon Script Execution**

Example:
```sh
    $ python src\train.py -s opsd --rmse
```
or
```sh
    $ python src\train.py -s opsd --smoothed_quantiles 0.025 0.975
```

| Parameter | Data Type  |  Value Range | Short Description |
|:---------:|:-------------:|:-------:|:-------------------:|
| --< loss > |  string in shell | {mse, mape, rmse, mis, nll_gauss, quantiles, smoothed_quantiles, crps} | Set the loss function for training |
| --ci |  boolean | True or False | Enables execution mode optimized for GitLab's CI |
| --logname |  str | " " | Name of the run, displayed in Tensorboard |

