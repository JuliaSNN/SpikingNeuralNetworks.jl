using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using SNNPlots
using Statistics
using Distributions

## Define the network parameters
NE = 1000
NI = 1000 ÷ 4
NI1 = round(Int, NI * 0.35)
NI2 = round(Int, NI * 0.65)

# Import models parameters
I1_params = duarte2019.PV
I2_params = duarte2019.SST
E_params = quaresima2022
@unpack connectivity, plasticity = quaresima2023


E = let
    @unpack dends, NMDA, param, soma_syn, dend_syn = E_params
    E = SNN.Tripod(
        dends...;
        N = NE,
        soma_syn = soma_syn,
        dend_syn = dend_syn,
        NMDA = NMDA,
        param = param,
    )
end


## Stimulus
# Background noise
stimuli = Dict(
    :noise_e =>
        SNN.PoissonStimulus(E, :he_s, param = 4.0kHz, neurons = :ALL, μ = 2.7f0),
    :noise_i =>
        SNN.PoissonStimulus(E, :hi_s, param = 1.0kHz, neurons = :ALL, μ = 3.0f0),
)
model = merge_models(stimuli, E = E)


# %%
SNN.monitor!(
    model.pop.E,
    [:fire, :v_d1, :v_s, :v_d1, :v_d2, :h_s, :h_d1, :h_d2, :g_d1, :g_d2],
)
SNN.train!(model = model, duration = 5s, pbar = true, dt = 0.125)
SNN.raster(model.pop, (4000, 5000))

## Target activation with stimuli
p = plot()
SNN.vecplot!(
    p,
    model.pop.E,
    :v_d1,
    r = 2.5s:4.5s,
    neurons = 1:1000,
    dt = 0.125,
    pop_average = true,
    ylims = :auto,
    ribbon = false,
)
SNN.vecplot!(
    p,
    model.pop.E,
    :v_d2,
    r = 2.5s:4.5s,
    neurons = 1:1000,
    dt = 0.125,
    pop_average = true,
    ylims = :auto,
    ribbon = false,
)
SNN.vecplot!(
    p,
    model.pop.E,
    :v_s,
    r = 2.5s:4.5s,
    neurons = 1:1000,
    dt = 0.125,
    pop_average = true,
    ylims = :auto,
    ribbon = false,
)
plot!(ylims = :auto)
