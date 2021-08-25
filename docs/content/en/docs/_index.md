
---
title: "Documentation"
linkTitle: "Documentation"
weight: 20
menu:
  main:
    weight: 20
---

ProLoaF is a **Pro**babilistic **Loa**d **F**orecasting Project and was intiially developed to support transmission and distribution system operators of electrical grids in their operational planning. The first use cases were developed in the context of the H2020 EU CoordiNet Project to better align transmission and distribution system operation in the light of growing intermittency in the power generation and consumption. These have particularly changed in the past years on the end consumers end, which has raised the need to pipeline and accelerate tools, such as forecasting to support a secure grid operation. 

ProLoaF makes use of the big data paradigm that allows machine learning algorithms, trained with data from the power system field. The core comprises a recurrent neural network (encoder-decoder architecture) to predict the target variable. 
The targets can vary from timerseries of PV, Wind or other generators to most commonly the total energy consumption.
Both, the relevant input data history and prediction horizon are arbitrarily long and customizable for any specific need.
