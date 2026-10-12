using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

# A new stimulus model: Poisson input with an absolute refractory period. A new parameter type
# (subtype of `PoissonStimulusParameter`) and its `stimulate!` method are enough: the existing
# `PoissonStimulus` container and its `Stimulus` constructor are reused.
@eval SNN.SNNModels begin
    @snn_kw struct PoissonRefractory{VFT = Vector{Float32}} <: PoissonStimulusParameter
        rate::Float32 = 10Hz
        ΔT::Float32 = 2ms                       # absolute refractory period
        μ::Float32 = 1.0f0                      # increment per spike
        N::Int = 100                            # size of the target population
        last_spike::VFT = fill(-Inf32, N)
        active::VBT = [true]
    end

    function stimulate!(p::PoissonStimulus, param::PoissonRefractory, time::Time, dt::AbstractFloat)
        param.active[1] || return
        @unpack rate, ΔT, μ, last_spike = param
        t = get_time(time)
        @inbounds for n in p.neurons
            if t - last_spike[n] > ΔT && rand() < rate * dt
                p.g[n] += μ
                last_spike[n] = t
            end
        end
    end
end

# An Identity neuron fires in every step in which it receives input: it relays the stimulus spikes
neuron = SNN.Identity(N = 1, name = "Identity Neuron")
stim = SNN.Stimulus(SNN.SNNModels.PoissonRefractory(N = 1, rate = 100Hz, ΔT = 20ms), neuron, :g)
SNN.monitor!(neuron, [:fire])
model = SNN.compose(; neuron, stim, silent = true)
SNN.sim!(; model, duration = 20s)

isi = diff(SNN.spiketimes(neuron)[1])
@assert minimum(isi) > 20ms
fig = Figure(size = (500, 350))
hist!(Axis(fig[1, 1], xlabel = "Inter-spike interval (ms)", ylabel = "Count"), isi, bins = 100)
save(joinpath(mktempdir(), "stimulus_model.png"), fig)
