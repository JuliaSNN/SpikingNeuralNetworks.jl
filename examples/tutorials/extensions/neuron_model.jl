using SpikingNeuralNetworks
using Distributions
using CairoMakie
SNN.@load_units

# A new population model: current-based leaky integrate-and-fire neurons. `@eval SNN.SNNModels`
# defines it inside SNNModels, as if it were a file of `SNNModels.jl/src/populations`.
@eval SNN.SNNModels begin
    """
        NeuronParameter

    Parameters of a current-based leaky integrate-and-fire neuron, used by `integrate!`.
    """
    NeuronParameter

    @snn_kw struct NeuronParameter <: AbstractPopulationParameter
        R::Float32 = 1GΩ
        El::Float32 = -70.6mV
        Vt::Float32 = -50.4mV
        Vr::Float32 = -70.6mV
        τm::Float32 = 20ms
        τe::Float32 = 10ms
        τi::Float32 = 10ms
    end

    """
        Neuron

    `N`, `param`, `name`, `id` and `records` are compulsory; `fire` is needed to record spikes and
    to project them through a `SpikingSynapse`. Vector types are type parameters (`@snn_kw` does
    not accept `Vector{Float32}` written in a field type).
    """
    Neuron

    @snn_kw struct Neuron{VFT = Vector{Float32}} <: AbstractPopulation
        param::NeuronParameter = NeuronParameter()
        N::Int = 10
        name::String = "Neuron"
        id::String = randstring(12)
        v::VFT = fill(param.El, N)
        ge::VFT = zeros(Float32, N)
        gi::VFT = zeros(Float32, N)
        fire::VBT = zeros(Bool, N)
        I::VFT = zeros(Float32, N)
        records::Dict = Dict()
    end

    # one forward Euler step; the recordings are handled by the simulation loop
    function integrate!(p::Neuron, param::NeuronParameter, dt::AbstractFloat)
        @unpack N, v, ge, gi, fire, I = p
        @unpack R, El, Vt, Vr, τm, τe, τi = param
        @inbounds for i in 1:N
            v[i] += dt / τm * (El - v[i] + R * (ge[i] - gi[i] + I[i]))
            ge[i] -= dt * ge[i] / τe
            gi[i] -= dt * gi[i] / τi
            fire[i] = v[i] >= Vt
            fire[i] && (v[i] = Vr)
        end
    end

    # where connections and spiking stimuli deliver their input
    function synaptic_target(targets::Dict, post::Neuron, sym::Symbol, target = nothing)
        push!(targets, :sym => sym)
        return getfield(post, sym), post.v
    end
end

neuron = SNN.SNNModels.Neuron(N = 1)
current_param = SNN.CurrentNoise(neuron.N; I_base = 20pA, I_dist = Normal(0pA, 20pA))
current_stim = SNN.Stimulus(current_param, neuron, :I)
input = SNN.Stimulus(SNN.PoissonLayer(rate = 10Hz, N = 100), neuron, :ge; conn = (p = 1, μ = 1pA))

SNN.monitor!(neuron, [:v, :fire, :ge, :I], sr = 2kHz)
model = SNN.compose(; neuron, current_stim, input, silent = true)
SNN.sim!(; model, duration = 1000ms)

fig = Figure(size = (600, 500))
SNN.vecplot!(Axis(fig[1, 1], ylabel = "Membrane potential (mV)"), neuron, :v, add_spikes = true)
SNN.vecplot!(Axis(fig[2, 1], ylabel = "Input current (pA)", xlabel = "Time (s)"), neuron, :I)
save(joinpath(mktempdir(), "neuron_model.png"), fig)
