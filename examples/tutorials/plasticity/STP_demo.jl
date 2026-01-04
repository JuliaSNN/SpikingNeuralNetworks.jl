using SpikingNeuralNetworks
SNN.@load_units

neuron = SNN.IF(; N = 1)
input = SNN.Identity(; N = 1)
stp_param = SNN.MarkramSTPParameterEvent(τD = 200ms, τF = 2000ms, U = 0.26)
syn = SNN.SpikingSynapse(input, neuron, :ge; conn=(μ = 1, p = 1), STPParam = stp_param)

model = SNN.compose(; neuron, input, syn)
SNN.train!(model, duration = 3s)
SNN.reset_time!(model)
SNN.monitor!(syn, :ρ)

interval = 50ms
spiketimes = [10s:interval:10400ms |> collect]
push!(spiketimes[1], 10520ms)
spike_param = SNN.SpikeTimeParameter(spiketimes)
spikes = SNN.SpikeTimeStimulus(input, :g, param = deepcopy(spike_param), conn=(μ = 1, p = 1))
SNN.update_spikes!(spikes, spike_param)

# SNN.monitor!(syn, :ρ)
SNN.monitor!([neuron], [:fire])
SNN.monitor!(syn, [:x, :u], variables = :STPVars)
SNN.monitor!([neuron], [:ge], variables = :synvars)
SNN.monitor!([syn], [:ρ])
model = SNN.compose(; neuron, input, syn, spikes)
SNN.train!(model, duration = 13s)

ge = SNN.record(neuron, :synvars_ge, interval = 9.8s:11s)
x = SNN.record(syn, :STPVars_x, interval = 9.8s:11s)
u = SNN.record(syn, :STPVars_u, interval = 9.8s:11s)
SNN.vecplot(syn, :ρ ,interval=9.8s:11s)
plot()
plot!(x(1, 9.901s:10.9s), label = "x", color = :blue)
plot!(u(1, 9.901s:10.9s), label = "u", color = :red)
plot!(ge(1, 9.901s:10.9s), label = "ge", color = :green)
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