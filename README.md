<div align="center">
    <img src="docs/src/assets/SNNLogo.svg" alt="SpikingNeuralNetworks.jl" width="200">
</div>

<h2 align="center">A Spiking Neural Network framework for Julia
<p align="center">
    <a href="https://github.com/JuliaSNN/SpikingNeuralNetworks.jl/actions">
    <img src="https://github.com/JuliaSNN/SpikingNeuralNetworks.jl/workflows/CI/badge.svg"
         alt="Build Status">
  </a>
  <a href="https://juliasnn.github.io/SpikingNeuralNetworks.jl/dev/">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg"
         alt="stable documentation">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yelllow"
       alt="bibtex">
  </a>

</p>
</h2>

# SpikingNeuralNetworks.jl Documentation

Julia Spiking Neural Networks (JuliaSNN) is a library for simulating biophysical neuronal network models. It provides models, plots, and analysis functions for spiking and rate networks simulations.


## Simple and powerful simulation framework

The library's strength points are:
 - Modular, intuitive, and quick instantiation of complex biophysical models;
 - Large pool of standard models already available and easy implementation of custom new models;
 - High performance and native multi-threading support, laptop and cluster-friendly;
 - Access to all models' variables at runtime and save-load-rerun of arbitrarily complex networks;
 - Growing ecosystem for stimulation protocols, network analysis, and visualization ([SNNUtils](https://github.com/JuliaSNN/SNNUtils), [SNNPlots](https://github.com/JuliaSNN/SNNPlots), [SNNGeometry](https://github.com/JuliaSNN/SNNGeometry)).

`SpikingNeuralNetworks.jl` leverages the `JuliaSNN` ecosystem, which offers `SNNPlots` to plot models' recordings and `SNNUtils` for further stimulation protocols and analysis.


SpikingNeuralNetworks.jl is a toolbox written in Julia 
The SpikingNeuralNetwork.jl package is an umbrella package for:
- `SNNModels.jl`
- `SNNPlots.jl`
- `SNNUtils.jl`
and future contributions of the spiking neural network ecosystem.

**Documentation** is [here](https://juliasnn.github.io/SpikingNeuralNetworks.jl/dev/).


