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
SNNPlots.@plots_default

ASSET_PATH = joinpath(@__DIR__, "../..", "docs", "src", "assets", "examples")