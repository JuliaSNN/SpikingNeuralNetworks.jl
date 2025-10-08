using SpikingNeuralNetworks
SNN.@load_units
using Statistics, Random, StatsPlots


## Classical STDP learning rule from:
## Gerstner, W., Kempter, R., van Hemmen, J. L., & Wagner, H. (1996). A neuronal learning rule for sub-millisecond temporal coding. Nature, 383(6595), 76–78. https://doi.org/10.1038/383076a0

stdp_param = SNN.STDPGerstner(
    A_pre = 5e-2,
    A_post = -5e-2,
    τpre = 15ms,
    τpost = 25ms,
    Wmax = Inf,
    Wmin = -Inf,
)
p1 = SNN.stdp_kernel(stdp_param, title = "Gerstner STDP", fill = true)


## Mexican Hat STDP
stdp_param = SNN.STDPMexicanHat(A = 2e-1, τ = 25ms, Wmax = Inf, Wmin = -Inf)
p2 = SNN.stdp_kernel(stdp_param, fill = true, title = "MexicanHat STDP")

## Structured connectivity with rate correction
stdp_param = SNN.STDPSymmetric(αpre = 0.0f0, A_x = 1, A_y = 1, Wmax = Inf, Wmin = -Inf)
p3 = SNN.stdp_kernel(
    stdp_param,
    fill = true,
    ΔTs = -1002.5:10:1002.5ms,
    title = "Symmetric Structured STDP",
)

stdp_param = SNN.STDPAntiSymmetric(
    αpre = 0.0f0,
    αpost = 0,
    A_x = 1,
    A_y = 1,
    Wmax = Inf,
    Wmin = -Inf,
)
p4 = SNN.stdp_kernel(
    stdp_param,
    fill = true,
    ΔTs = -1002.5:10:1002.5ms,
    title = "Anti Symmetric Structured STDP",
)
# SNN.stdp_weight_decorrelated(stdp_param)

plot(p1, p2, p3, p4, layout = (2, 2), size = (900, 800), legend = false, margin = Plots.mm)

## 
stdp_param = SNN.STDPGerstner(
    A_pre = 1,
    A_post = -1,
    τpre = 20ms,
    τpost = 20ms,
    Wmax = Inf,
    Wmin = -Inf,
)
SNN.stdp_kernel(stdp_param, fill = true, lw = 0)

rates = 0:2Hz:30Hz
ΔWs = zeros(length(rates), length(rates))
for n in eachindex(rates)
    Threads.@threads for m in eachindex(rates)
        pre = rates[n]
        post = rates[m]
        w = SNN.stdp_weight_decorrelated(stdp_param, pre, post) |> mean
        ΔWs[m, n] = w
    end
end
heatmap(ΔWs)
##
