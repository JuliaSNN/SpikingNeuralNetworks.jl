using SpikingNeuralNetworks
using CairoMakie, Makie
SNN.@load_units

neuron = SNN.IF(; N = 1, synapse=SNN.SingleExpSynapse(τe = 5ms))
input = SNN.Identity(; N = 1)
stp_param = SNN.MarkramSTPParameterHet(τD = [200ms], τF = [200ms], U = [0.4])
# stp_param = SNN.MarkramSTPParameter(τD = 200ms, τF = 200ms, U = 0.4)
# stp_param = SNN.MarkramSTPParameterTimestep(τD = 200ms, τF = 200ms, U = 0.4)
syn = SNN.SpikingSynapse(input, neuron, :ge; conn=(μ = 1, p = 1), STPParam = stp_param)

model = SNN.compose(; neuron, input, syn)
SNN.train!(model, duration = 3s)
SNN.reset_time!(model)
SNN.monitor!(syn, :ρ)

interval = 50ms
spiketimes = [10s:interval:10400ms |> collect]
push!(spiketimes[1], 10520ms)
spike_param = SNN.SpikeTimeParameter(spiketimes)
spikes = SNN.SpikeTimeStimulus(input, :g, param = deepcopy(spike_param), conn=(μ = 2, p = 1))
SNN.update_spikes!(spikes, spike_param)

# SNN.monitor!(syn, :ρ)
SNN.monitor!([neuron], [:fire], sr=2kHz)
SNN.monitor!(syn, [:x, :u], variables = :STPVars, sr=2kHz)
SNN.monitor!([neuron], [:ge, :glu], variables = :synvars, sr=2kHz)
SNN.monitor!([syn], [:ρ], sr=2kHz)
model = SNN.compose(; neuron, input, syn, spikes)
SNN.train!(model, duration = 13s)

ge = SNN.record(neuron, :synvars_ge, interval = 9.8s:11s)

x = SNN.record(syn, :STPVars_x, interval = 9.8s:11s)
u = SNN.record(syn, :STPVars_u, interval = 9.8s:11s)

fig = Figure(size=(400,300))
ax = Axis(fig[1,1], xlabel = "Time (s)", ylabel = "Value", title = "STP Dynamics")
SNN.vecplot!(ax, syn, :ρ ,interval=9.8s:11s, color=:black)

interval = 9.901s:0.1:10.9s
lines!(ax, interval, x(1, interval), label = "x", color = :blue, alpha=0.7)
lines!(ax, interval, u(1, interval), label = "u", color = :red, alpha=0.7)
lines!(ax, interval, ge(1,interval), label = "ge", color = :green, linewidth=3)
axislegend(ax, position = :rt)
fig
##
using SpikingNeuralNetworks

function test_STP(; U, τF, τD, interval = 50ms)
    neuron = SNN.IF(; N = 1)
    input = SNN.Identity(; N = 1)
    stp_param = SNN.MarkramSTPParameter(; τD, τF, U)
    syn = SNN.SpikingSynapse(input, neuron, :ge; conn=(μ = 1, p = 1), STPParam = stp_param)

    spiketimes = [10s:interval:10400ms |> collect]
    push!(spiketimes[1], 10520ms)
    spike_param = SNN.SpikeTimeParameter(spiketimes)
    spikes = SNN.SpikeTimeStimulus(input, :g, param = deepcopy(spike_param), conn=( p = 1, μ = 1))
     SNN.update_spikes!(spikes, spike_param)

    # SNN.monitor!(syn, :ρ)
    SNN.monitor!([neuron], [:fire])
    SNN.monitor!([neuron], [:ge], variables = :synvars)
    model = SNN.compose(; neuron, input, syn, spikes)
    SNN.train!(model, duration = 3s, pbar=true)

    # ge = SNN.record(neuron, :synvars_ge, interval = 9.8s:11s)
    # return ge(1, 9.901s:10.9s)
end

@profview test_STP(U = 0.0125, τF = 499ms, τD = 200ms, interval = 50ms) #|> plot