# Firing patterns of the AdEx neuron under a current step (Naud et al. 2008 parameter sets).
using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

# (pattern, τm (ms), a (nS), τw (ms), b (pA), Vr (mV), I (pA))
patterns = [
    ("Tonic", 20, 0.0, 30.0, 60.0, -55.0, 65),
    ("Adapting", 20, 0.0, 100.0, 5.0, -55.0, 65),
    ("Init. burst", 5.0, 0.5, 100.0, 7.0, -51.0, 65),
    ("Bursting", 5.0, -0.5, 100.0, 7.0, -46.0, 65),
    ("Transient", 10, 1.0, 100, 10.0, -60.0, 65),
    ("Delayed", 5.0, -1.0, 100.0, 10.0, -60.0, 25),
]

fig = Figure(size = (1200, 900))
for (n, (name, τm, a, τw, b, Vr, I)) in enumerate(patterns)
    param = SNN.AdExParameter(R = 0.5GΩ, Vt = -50mV, ΔT = 2mV, El = -70mV, τm = τm * ms,
                              Vr = Vr * mV, a = a * nS, b = b * pA, τw = τw * ms)
    E = SNN.AdEx(; N = 1, param)
    SNN.monitor!(E, [:v, :fire, :w], sr = 8kHz)
    model = SNN.compose(; E, silent = true)
    E.I .= 5pA                      # weak current for 30 ms
    SNN.sim!(; model, duration = 30ms)
    E.I .= I * pA                   # current step
    SNN.sim!(; model, duration = 300ms)

    row, col = (n - 1) ÷ 2 + 1, 2 * ((n - 1) % 2)
    ax1 = Axis(fig[row, col + 1], title = name, ylabel = "V (mV)")
    SNN.vecplot!(ax1, E, :v, add_spikes = true, color = :black)
    ax2 = Axis(fig[row, col + 2], ylabel = "w (pA)")
    SNN.vecplot!(ax2, E, :w, color = :grey)
end
save(joinpath(mktempdir(), "AdEx_neuron_types.png"), fig)
