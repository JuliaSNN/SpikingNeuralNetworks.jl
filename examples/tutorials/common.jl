using SpikingNeuralNetworks
using SNNPlots
import SNNPlots: vecplot, gplot

SNN.@load_units

using Plots
using Distributions
using UnPack
using Random
ASSET_PATH = joinpath(@__DIR__, "../..", "docs", "src", "assets", "examples")
Plots.default(palette = :okabe_ito)
Plots.default(size = (800, 600))
