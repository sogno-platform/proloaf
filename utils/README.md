# plf-util
This package supports building, logging and managing machine learning models, and is designed for timeseries problems using [Pytorch](https://pytorch.org/).

- The datatuner supports managing and scaling of any input for machine learning.
- plf-util comes with functions to define and apply scores for the training of neural networks (Seq-2-Seq RNN) which shall predict the near-time future. In this context probabilistic metrics are collected, both to be used in training and ex-post testing.
- With config-util we've created a support to design and modify the RNN via config files only.  

## License
This project is licensed under the GNU General Public License v3.0.