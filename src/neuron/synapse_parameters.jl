## Soma synapse parameters

EyalNMDA = let
    Mg_mM = 1.0mM
    nmda_b = 3.36   # voltage dependence of nmda channels
    nmda_k = -0.077     # Eyal 2018
    NMDAVoltageDependency(mg = Mg_mM/mM, b = nmda_b, k = nmda_k)
end

## Tripod
MilesGabaSoma =
    GABAergic(Receptor(E_rev = -70.0, τr = 0.1, τd = 15.0, g0 = 0.38), Receptor())
DuarteGluSoma = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
    ReceptorVoltage(E_rev = 0.0, nmda = 0.0f0),
)
EyalGluDend = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
    ReceptorVoltage(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 1.31, nmda = 1.0f0),
)
MilesGabaDend = GABAergic(
    Receptor(E_rev = -70.0, τr = 4.8, τd = 29.0, g0 = 0.27),
    Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.006), # τd = 100.0
)

TripodSomaSynapse = Synapse(DuarteGluSoma, MilesGabaSoma)
TripodDendSynapse = Synapse(EyalGluDend, MilesGabaDend)


## Soma parameters

SomaNMDA = let
    Mg_mM = 1.0mM
    nmda_b = 3.57   # voltage dependence of nmda channels
    nmda_k = -0.062     # Eyal 2018
    NMDAVoltageDependency(mg = Mg_mM/mM, b = nmda_b, k = nmda_k)
end
SomaGlu = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 1ms, τd = 6.0ms, g0 = 0.7),
    ReceptorVoltage(E_rev = 0.0, τr = 1ms, τd = 100.0, g0 = 0.15, nmda = 1.0f0),
)
SomaGABA = GABAergic(
    Receptor(E_rev = -70.0, τr = 0.5, τd = 10.0, g0 = 2.0),
    Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.006), # τd = 100.0
)
SomaNMDA = NMDAVoltageDependency()
SomaSynapse = Synapse(SomaGlu, SomaGABA)

## CAN_AHP parameters

Glu_CANAHP = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 1, τd = 2.5ms, g0 = 0.2mS/cm^2),
    ReceptorVoltage(E_rev = 0.0, τr = 4.65ms, τd = 75ms, g0 = 0.3mS/cm^2, nmda = 1.0f0),
)
Gaba_CANAHP = GABAergic(
    Receptor(E_rev = -70.0, τr = 1, τd = 10ms, g0 = 0.35mS/cm^2),
    Receptor(E_rev = -90.0, τr = 90ms, τd = 160ms, g0 = 5e-4mS/cm^2), # τd = 100.0
)
Synapse_CANAHP = Synapse(Glu_CANAHP, Gaba_CANAHP)
αs_CANAHP = [1.0, 0.275/ms, 1.0, 0.015/ms]
NMDA_CANAHP  = let
    Mg_mM = 1.5mM
    nmda_b = 3.57
    nmda_k = -0.063
    NMDAVoltageDependency(mg = Mg_mM/mM, b = nmda_b, k = nmda_k)
end


export SomaNMDA,
    SomaSynapse, TripodSomaSynapse, TripodDendSynapse, EyalNMDA, NMDA_CANAHP, Synapse_CANAHP
