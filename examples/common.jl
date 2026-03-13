using DrWatson
findproject() |> quickactivate  

using Distributions
using UnPack
using Random
using Statistics
using StatsBase
using LaTeXStrings

using SpikingNeuralNetworks
import SNNPlots: vecplot, gplot
using SNNPlots
using SNNUtils

SNN.@load_units
SNNPlots.@makie_default

ASSET_PATH = joinpath(@__DIR__, "..", "docs", "src", "assets", "examples")
@assert isdir(ASSET_PATH) "Asset path does not exist: $ASSET_PATH"