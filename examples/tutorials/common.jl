using SpikingNeuralNetworks
using SNNPlots
import SNNPlots: vecplot
using Plots
@load_units
ASSET_PATH = joinpath(@__DIR__, "../..", "docs", "src", "assets", "examples")
Plots.default(palette = :okabe_ito)