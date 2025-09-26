using SpikingNeuralNetworks
SNN.@load_units

# Passive neuron parameters, no NMDA receptor
passive_neuron = SNN.DendNeuronParameter(
    dend_syn = SNN.SynapseArray(
        [SNN.Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 2.73),
        SNN.ReceptorVoltage(),
        SNN.Receptor(),
        SNN.Receptor()
        ]
),
    ds = [150um],
)

# Active neuron with NMDA receptor
active_neuron = SNN.DendNeuronParameter(
    dend_syn = SNN.SynapseArray(
        [SNN.Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
        SNN.Receptor(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 1.31, nmda = 1.0f0),
        SNN.Receptor(),
        SNN.Receptor()
        ]
    ),
    ds = [150um],
)

# Populations of single-dendrite neurons, one with active dendrites, one with passive dendrites
E_active = SNN.BallAndStick(N=200; name="Active", param =  active_neuron)
E_passive = SNN.BallAndStick(N=200; name="Passive", param = passive_neuron)

# Input projections
input = SNN.Poisson(N=100, param=SNN.PoissonParameter(rate=20Hz), name="Input")
input_passive = SNN.SpikingSynapse(input, E_passive, :glu, :d; μ = 2, p = 0.5)
input_active = SNN.SpikingSynapse(input, E_active, :glu, :d; μ = 2, p = 0.5)

# Recurrent connections
EE_passive = SNN.SpikingSynapse(E_passive, E_passive, :glu, :d; μ = 2, p = 0.01)
EE_active = SNN.SpikingSynapse(E_active, E_active, :glu, :d; μ = 2, p = 0.01)

# Compose the model
model = SNN.compose(; E_active, E_passive, input,  input_passive, input_active, EE_active, EE_passive)

# Simulate the model
SNN.monitor!(model.pop, [:fire])
for n in 1:5
    model.pop.input.rate .= 25Hz
    SNN.sim!(model, duration = 0.05second)
    model.pop.input.rate .= 0Hz
    SNN.sim!(model, duration = 0.05second)
end

SNN.raster(model.pop, [0.1second, 0.5second], yrotation=90)
