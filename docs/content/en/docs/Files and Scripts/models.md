### The Models
There are multiple models available to be created using ProLoaF. 

#### Recurrent Neural Networks (RNN)
The default recommended model is an [long-short-term-memory (LSTM)](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html). This model also allows the use of other recurrent model like [gated-recurrent-units (GRU)](https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.dynamic.GRU.html).
The models are dynamically imported in the constructor so any model that implements `torch.nn.Module` and follows the signature of LSTM and GRU can be used as core model. 
<!-- TODO add description of input output Format -->

#### Transformer Models
A [Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#transformer) model is available for forecasting. This model could not show superior results when compared to RNN models while using a higher amount of resources and training time. Since theses test were somewhat limited the model could be reevaluated for a specific use-case, specifically when complex temporal dependencies are suspected and the size of the training dataset is sufficently large. 
<!-- TODO possible to reference Master thesis? -->

#### AutoEncoder Hybrid Model
This is an experimental model not recommended for practial use. This model is based on the RNN model, but combines the forecast with a decoder for recustruction of the input data during training, as would be the case in an autoencoder. 
<!-- TODO add reference to the original paper about this hybrid model -->