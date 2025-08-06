using Plots
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter
using Statistics, Random
##

ν = 4.5 * 1000Hz
N = 1000
p_in = 0.05
μ_in = 7

ge_in = μ_in * ν * (N * p_in) * 3 * AdExParameter().τde
R = SNN.C / SNN.gL


## LKD parameters
τm = 20ms
C = 300SNN.pF
R = τm / C

LKD_AdEx_exc = AdExParameterSingleExponential(
    τm = 20ms,
    τe = 6ms,
    τi = 2ms,
    Vt = -52mV,
    Vr = -60mV,
    El = -70mV,
    R = R,
)
LKD_AdEx_inh = AdExParameterSingleExponential(
    τm = 20ms,
    τe = 6ms,
    τi = 2ms,
    Vt = -52mV,
    Vr = -60mV,
    El = -62mV,
    R = R,
)

# inputs, the kHz is obtained by the N*ν, so doing the spikes (read eqs. 4 section)
N = 1000
νe = 4.5Hz
νi = 2.5Hz
p_in = 1.0
μ_in = 1.50
C = 300SNN.pF

μEE = 2.76 / C * SNN.pF
μEI = 1.27 / C * SNN.pF
μIE = 48.7 / C * SNN.pF
μII = 16.2 / C * SNN.pF


#
E = SNN.AdEx(; N = 4000, param = LKD_AdEx_exc)
I = SNN.AdEx(; N = 1000, param = LKD_AdEx_inh)

EE = SNN.SpikingSynapse(E, E, :ge; μ = μEE, p = 0.2)
EI = SNN.SpikingSynapse(E, I, :ge; μ = μEI, p = 0.2)
IE = SNN.SpikingSynapse(I, E, :gi; μ = μIE, p = 0.2)
II = SNN.SpikingSynapse(I, I, :gi; μ = μII, p = 0.2)

Input_E = SNN.PoissonStimulus(E, :ge, param = νe, neurons = :ALL)
Input_I = SNN.PoissonStimulus(I, :ge, param = νi, neurons = :ALL)

##
model = compose(SNN.@symdict E I Input_E Input_I EE EI IE II)
SNN.monitor(model.pop, [:fire])

SNN.sim!(P, C; duration = 15second)
SNN.raster(P, [4.3s, 5s])
SNN.monitor([E, I], [:ge, :gi, :v])
SNN.sim!(P, C; duration = 1second)

plot([hcat(E.records[:ge]...)[123, :], hcat(E.records[:gi]...)[123, :]])
