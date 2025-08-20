module SpikingNeuralNetworks


using SNNModels
using SNNPlots
using SNNUtils

export SNNModels, SNNPlots, SNNUtils

SNN = SpikingNeuralNetworks
export SNN

DOCS_ASSETS_PATH = joinpath(dirname(dirname(pathof(SpikingNeuralNetworks))), "docs", "src", "assets") 
export DOCS_ASSETS_PATH


end
