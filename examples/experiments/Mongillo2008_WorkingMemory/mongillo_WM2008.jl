import SNNModels: DeltaSynapse, IF
function Mongillo2008(; n_assemblies = 1, n_neurons = 800)
    MongilloParam = (
        Exc = IFParameter(
            R = 1,
            τm = 15ms,
            Vt = 20mV,
            Vr = 16mV,
            El = 0mV,
            τabs = 2ms,
        ),
        Inh = IFParameter(
            R = 1,
            τm = 10ms,
            Vt = 20mV,
            El = 0mV,
            Vr = 13mV,
        ),
        spike = (;
            At = 0ms,
            τabs = 2ms,
        ),
        synapse = DeltaSynapse(),

    )
    @unpack Exc, Inh, spike, synapse = MongilloParam
    pop = (
        E = IF(;N = 8000, param = Exc, spike, synapse),
        I = IF(;N = 2000, param = Inh, spike, synapse),
    )

    connections = (;
        EE = (p = 0.2, σ = 0, μ = 0.10 * 8000/pop.E.N),
        EI = (p = 0.2, σ = 0, μ = 0.135 * 8000/pop.E.N),
        IE = (p = 0.2, σ = 0, μ = 0.25 * 2000/pop.I.N),
        II = (p = 0.2, σ = 0, μ = 0.20 * 2000/pop.I.N),
    )

    input_exc = 19.8
    input_inh = 19.8

    conn 
    syn = (
        EE = SpikingSynapse(
            pop.E,
            pop.E,
            :ge,
            conn = connections.EE,
            param = SNN.STPParameter(),
            delay_dist = Uniform(1ms, 5ms),
        ),
        EI = SpikingSynapse(
            pop.E,
            pop.I,
            :ge,
            conn = connections.EI,
            delay_dist = Uniform(1ms, 5ms),
        ),
        IE = SpikingSynapse(
            pop.I,
            pop.E,
            :gi,
            conn = connections.IE,
            delay_dist = Uniform(1ms, 5ms),
        ),
        II = SpikingSynapse(
            pop.I,
            pop.I,
            :gi,
            conn = connections.II,
            delay_dist = Uniform(1ms, 5ms),
        ),
    )

    stim = (
        E = SNN.CurrentStimulus(pop.E, I_dist = Normal(input_exc, 1.0), α = 1.0),
        I = SNN.CurrentStimulus(pop.I, I_dist = Normal(input_inh, 1.0), α = 1.0),
    )

    μee_assembly = 0.48 * 8000/pop.E.N
    model = SNN.compose(pop, syn, stim)
    assemblies = map(1:n_assemblies) do x
        neurons = StatsBase.sample(1:pop.E.N, n_neurons, replace = false)
        update_weights!(syn.EE, neurons, neurons, μee_assembly)
        (
            neurons = neurons,
            name = Symbol("assembly$x"),
            indices = indices(syn.EE, neurons, neurons),
            id = x,
        )
    end
    return model, assemblies
end
