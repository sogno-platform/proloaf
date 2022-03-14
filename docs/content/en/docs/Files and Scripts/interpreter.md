---
title: 'interpreter.py'
linkTitle: 'interpreter.py'
date: 2022-02-16
weight: 3
description: >
    Basic information on the interpretation script.
---

### Overview

This script evaluates a trained neural network on its salient features regarding the time and feature dimension and creates a saliency heatmap.
The neural network should be trained beforehand.

### Inputs

-   ./data/<path_to_data_csv>.csv
-   ./oracles/model_station

### Outputs

# Plots, tensors and metrics in structures table

-   ./oracles/interpretation/<name_of_evaluation_folder>/

### Example Output Files

The output consists of the most salient features visualized in the form of saliency maps split up into multiple .png images together with a csv file on the metrics and the model's output tensors.
![Basic Saliency Map]({{< resource url="img/heatmap_HL_base.png" >}} "Basic Saliency Map")_Basic saliency map with 7 days long recent history horizon input (predictor variable load, and hour of the day) in the encoder and one day prediction. Decoder inputs are sinus and cosinus encoding of the hour of the day._

![Larger Saliency Study]({{< resource url="img/heatmap_HL_plus_zoomed.png" >}} "Larger Saliency Study")_Saliency map showing the importance of features per timestep tp maintain the trained loss._

### Step-by-step guide what the interpreter does:

#### Step 1: Preparations

Select and download the data on which you want to train a neural network on and put it in the data folder.
Make a new folder in the targets folder and put a `config.json` and tuning.json file within.
There you can set all the relevant paths and parameters.
Now train the neural network on the data by using the fc_train function.
The resulting net will be saved in the oracles folder

Now in `__main__` function of the `interpreter.py` the following setting need to be made:

-   set the `MAX_BATCH_SIZE` to the number of references to use for each feature (recommended: At least 10)
-   set `MAX_EPOCHS` to the number of epochs of the saliency map optimization process (recommended: At least a few thousand)
-   set `N_TRIALS` to the number of hyperparameter search trials to run (recommended: at least 30)
-   set `MODEL_NAME` to the name of the folder in the targets folder created before
-   set `SEP` to the column seperation sign used in the data file (eg. ";" or ",")
-   set `timesteps` to any number of timesteps to interpret (e.g [1,2,3])
    > beware that the timestep indicates the beginning of the history horizon, not the forecast horizon.
    > Also the timestep should be identical to the one of the internal dataloader not the datasheet file in the data folder

#### Step 2: Run the Function

Now run the `interpreter.py` script.
The function will now read in all the parameters it needs from the `config.json` file.
The trained net will be loaded and the saliency map is calculated for each timestep.
During the hyperparameter search some tensor will be saved to the oracles/interpretation/Tensors folder
and the Tensors of the best performing saliency map are loaded again automatically after the process for plotting purposes.

#### Step 3: Looking at the Results

After the `interpreter.py` script has finished several plots can be found in the oracles/interpretation/plots/\<model name\> folder:

-   plot of the saliency map
-   the corresponding input feature plots
-   a plot with the target values, the original predictions and the perturbated predictions

The numbers in the plot names refer to the timestep of the plot
