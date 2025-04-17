using SNNPlots
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter
using Statistics, Random, StatsSNNPlots


##
stdp_param = STDPParameter(A_pre = -5e-1, A_post = 5e-1, τpre = 20ms, τpost = 15ms)
SNN.stdp_kernel(stdp_param)
SNN.stdp_weight_decorrelated(stdp_param)
##

stdp_param = STDPParameter(A_pre = 5e-2, A_post = -5e-2, τpre = 15ms, τpost = 25ms)
SNN.stdp_kernel(stdp_param)
SNN.stdp_weight_decorrelated(stdp_param)
##

stdp_param = STDPMexicanHat(A = -2e-1, τ = 25ms)
SNN.stdp_kernel(stdp_param, fill = false)
SNN.stdp_weight_decorrelated(stdp_param)

##

stdp_param = iSTDPParameterTime()
SNN.stdp_kernel(stdp_param, fill = false)
SNN.stdp_weight_decorrelated(stdp_param)
##

stdp_param = SymmetricSTDP(αpre = 0.0f0, A_x = 0.1, A_y = 0.1)
p1 = SNN.stdp_kernel(stdp_param, fill = true, ΔTs = -970.5:10:1000ms)
SNN.stdp_weight_decorrelated(stdp_param)

stdp_param = AntiSymmetricSTDP(αpre = 0.0f0, A_x = 0.1, A_y = 0.1)
p2 = SNN.stdp_kernel(stdp_param, fill = true, ΔTs = -970.5:10:1000ms)
SNN.stdp_weight_decorrelated(stdp_param)

plot(p1, p2, layout = 2, size = (800, 400), legend = false, xlims = (-300, 300))
``
