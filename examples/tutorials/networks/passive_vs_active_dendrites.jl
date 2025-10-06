using SpikingNeuralNetworks
SNN.@load_units

## AdEx neuron with fixed external current connections with multiple receptors

passive_neuron = SNN.DendNeuronParameter(
    dend_syn = SNN.SynapseArray(
        [SNN.Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 2.73),
        SNN.ReceptorVoltage()]
),
    ds = [150um],
    gaba_receptors = []
)

active_neuron = SNN.DendNeuronParameter(
    dend_syn = SNN.SynapseArray(
        [SNN.Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
        SNN.Receptor(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 1.31, nmda = 1.0f0)]
    ),
    ds = [150um],
    gaba_receptors = []
)

E_active = SNN.BallAndStick(N=200; name="Active", param =  active_neuron)
E_passive = SNN.BallAndStick(N=200; name="Passive", param = passive_neuron)

input_param = SNN.PoissonLayer( N = 100, rate = 20Hz,)
projection = (p = 0.5,μ = 2nS)
I_active = SNN.Stimulus(input_param, E_active, :glu, :d, conn=projection)
I_passive =SNN.Stimulus(input_param, E_passive, :glu, :d, conn=projection)

EE_passive = SNN.SpikingSynapse(E_passive, E_passive, :glu, :d; conn= (μ = 2, p = 0.01))
EE_active = SNN.SpikingSynapse(E_active, E_active, :glu, :d; conn= (μ = 2, p = 0.01))

model = SNN.compose(; E_active, E_passive, I_active, I_passive, EE_active, EE_passive)
SNN.monitor!(model.pop, [:fire])

SNN.sim!(model, duration = 0.5second)
SNN.raster(model.pop, [0.1second, 0.5second], yrotation=90)