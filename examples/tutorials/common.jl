using SpikingNeuralNetworks
import SNNPlots: vecplot, gplot
using SNNPlots

SNN.@load_units
SNNPlots.@plots_default

using Distributions
using UnPack
using Random
ASSET_PATH = joinpath(@__DIR__, "../..", "docs", "src", "assets", "examples")