using SNNPlots
import SNNPlots: vecplot, plot
using SpikingNeuralNetworks
using DataFrames
SNN.@load_units

# Define the data
data = [
    ("Tonic", 20, 0.0, 30.0, 60.0, -55.0, 65),
    ("Adapting", 20, 0.0, 100.0, 5.0, -55.0, 65),
    ("Init. burst", 5.0, 0.5, 100.0, 7.0, -51.0, 65),
    ("Bursting", 5.0, -0.5, 100.0, 7.0, -46.0, 65),
    # ("Irregular", 14.4, -0.5, 100.0, 7.0, -46.0, 65),
    ("Transient", 10, 1.0, 100, 10.0, -60.0, 65),
    ("Delayed", 5.0, -1.0, 100.0, 10.0, -60.0, 25),
]

# Create the DataFrame
df = DataFrame(
    Type = [row[1] for row in data],
    τm = [row[2] for row in data],
    a = [row[3] for row in data],
    τw = [row[4] for row in data],
    b = [row[5] for row in data],
    ur = [row[6] for row in data],
    i = [row[7] for row in data],
)

# Display the DataFrame
println(df)

plots = map(eachrow(df)) do row
    param = AdExParameter(
        R = 0.5GΩ,
        Vt = -50mV,
        ΔT = 2mV,
        El = -70mV,
        # τabs=0,
        τm = row.τm * ms,
        Vr = row.ur * mV,
        a = row.a * nS,
        b = row.b * pA,
        τw = row.τw * ms,
        At = 0.0f0,
    )


    E = SNN.AdEx(; N = 1, param)
    SNN.monitor!(E, [:v, :fire, :w], sr = 8kHz)
    model = compose(; E = E, silent = true)

    E.I .= Float32(05pA)
    SNN.sim!(; model, duration = 30ms)
    E.I .= Float32(row.i)
    # E.I .= row.i, # Current step
    SNN.sim!(; model, duration = 300ms)

    default(color = :black)
    p1 = plot(
        vecplot(
            E,
            :v,
            add_spikes = true,
            ylabel = "Membrane potential (mV)",
            ylims = (-80, 10),
        ),
        vecplot(E, :w, ylabel = "Adapt. current (nA)", c = :grey, margin = 10Plots.mm),
        plot_title = row.Type,
        layout = (1, 2),
        legend = false,
        size = (600, 800),
        topmargin = 1Plots.mm,
    )
end

p = plot(
    plots...,
    layout = (3, 2),
    size = (1600, 1000),
    xlabel = "Time (ms)",
    leftmargin = 10Plots.mm,
)

savefig(
    p,
    "/home/user/mnt/zeus/User_folders/aquaresi/network_models/src/SpikingNeuralNetworks.jl/docs/src/assets/examples/AdEx.png",
)
