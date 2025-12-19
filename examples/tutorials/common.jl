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
Plots.default(size = (600, 400))
Plots.default(frame=:box,  
              grid=false, 
              tickfontsize=12, 
              guidefontsize=14, 
              legendfontsize=12, 
              margin=5Plots.mm, foreground_color_legend=:transparent, mscolor=:auto, lw=4)