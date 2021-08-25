---
title: About ProLoaF
linkTitle: About
menu:
  main:
    weight: 10

---


{{< blocks/cover title="About ProLoaF" image_anchor="bottom" height="min" >}}

{{< /blocks/cover >}}

{{% blocks/lead %}}
ProLoaF makes use of the big data paradigm that allows machine learning algorithms, trained with data from the power system field. The core comprises a recurrent neural network (encoder-decoder architecture) to predict the target variable. 
The targets can vary from timerseries of PV, Wind or other generators to most commonly the total energy consumption.
Both, the relevant input data history and prediction horizon are arbitrarily long and customizable for any specific need.

{{% /blocks/lead %}}

{{< blocks/lead color="dark" >}}
Here you can find instructions on [how to build and install ProLoaF]({{< resource url="docs/getting-started/" >}}) as
well as [descriptions of the scripts]({{< resource url="docs/files-and-scripts/" >}}) and
[tutorials]({{< resource url="docs/tutorials/" >}})  to get you started.
{{< /blocks/lead >}}
