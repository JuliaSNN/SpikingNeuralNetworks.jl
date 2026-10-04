using SpikingNeuralNetworks
SNN.@load_units
using CairoMakie

## STDP kernels of the plasticity rules.
##
## Weight change of one synapse for a single pre/post spike pair separated by ΔT = t_post - t_pre.
## The kernel is measured by feeding the spikes of the two neurons directly to `plasticity!`, so
## that exactly one pre and one post spike occur. (`SNN.stdp_kernel` drives the post neuron through
## the synapse itself, and the Identity post neuron then fires one step after the pre spike, which adds
## a constant LTP term A_pre to the kernel; the helper below avoids it.)
## Plasticity rules only act under `train!`; here `plasticity!` is called directly for the same reason.

function pair_kernel(param, ΔTs; dt = 0.1f0, t0 = 1100ms, w0 = 1.0f0)
    ΔWs = zeros(Float32, length(ΔTs))
    for (n, ΔT) in enumerate(ΔTs)
        pre, post = SNN.Identity(N = 1), SNN.Identity(N = 1)
        syn = SNN.SpikingSynapse(pre, post, :g; conn = fill(w0, 1, 1), LTPParam = param)
        T = SNN.Time()
        step_pre = round(Int, t0 / dt)
        step_post = step_pre + round(Int, ΔT / dt)
        for step = 1:(max(step_pre, step_post)+round(Int, 600ms / dt))
            pre.fire[1] = step == step_pre
            post.fire[1] = step == step_post
            SNN.update_time!(T, dt)
            SNN.plasticity!(syn, syn.param, dt, T)
        end
        ΔWs[n] = syn.W[1] - w0
    end
    return ΔWs
end

function kernel_axis!(fig, pos, param, ΔTs; title)
    ax = Axis(fig[pos...], xlabel = "ΔT = t_post - t_pre (ms)", ylabel = "ΔW", title = title)
    xs = Float64.(ΔTs)
    ΔWs = Float64.(pair_kernel(param, ΔTs))
    for sel in (xs .< 0, xs .> 0)
        lines!(ax, xs[sel], ΔWs[sel])
        band!(ax, xs[sel], zeros(count(sel)), ΔWs[sel], alpha = 0.3)
    end
    hlines!(ax, 0, color = :black, linewidth = 0.5)
    return ax
end

ΔTs = vcat(-100:2.5:-2.5, 2.5:2.5:100) .* ms

## Classical STDP learning rule from:
## Gerstner, W., Kempter, R., van Hemmen, J. L., & Wagner, H. (1996). A neuronal learning rule for
## sub-millisecond temporal coding. Nature, 383(6595), 76–78. https://doi.org/10.1038/383076a0
##
## Amplitudes are applied once and are signed (SNNModels >= 1.9):
##   ΔW =  A_pre  * exp(-ΔT / τpre)   for ΔT > 0  (pre before post, LTP, A_pre > 0)
##   ΔW =  A_post * exp( ΔT / τpost)   for ΔT < 0  (post before pre, LTD, A_post < 0)
## Values chosen for a classical asymmetric window (Bi & Poo 2001, with the time constants of
## Pfister & Gerstner 2006): peak LTP +1e-2, peak LTD -1.05e-2 (LTD slightly larger than LTP),
## τpre = 16.8 ms, τpost = 33.7 ms. The weight is 1, so the peaks are a 1 % change.
## (With the old A^2 semantics the former values A = ±5e-2 gave +2.5e-3 on both sides.)
stdp_gerstner = SNN.STDPGerstner(
    A_pre = 1e-2,
    A_post = -1.05e-2,
    τpre = 16.8ms,
    τpost = 33.7ms,
    Wmax = Inf,
    Wmin = -Inf,
)

fig = Figure(size = (900, 800))
ax1 = kernel_axis!(fig, (1, 1), stdp_gerstner, ΔTs; title = "Gerstner STDP")

## Check against the analytic kernel (the maximum deviation is Float32 rounding of the trace decay).
analytic(ΔT) = ΔT > 0 ? 1e-2 * exp(-ΔT / 16.8) : -1.05e-2 * exp(ΔT / 33.7)
@show maximum(abs.(pair_kernel(stdp_gerstner, ΔTs) .- analytic.(ΔTs)))

## Mexican hat STDP
stdp_mexican = SNN.STDPMexicanHat(A = 2e-1, τ = 25ms, Wmax = Inf, Wmin = -Inf)
ax2 = kernel_axis!(fig, (1, 2), stdp_mexican, ΔTs; title = "MexicanHat STDP")

## Structured connectivity with rate correction
stdp_sym = SNN.STDPSymmetric(αpre = 0.0f0, A_x = 1, A_y = 1, Wmax = Inf, Wmin = -Inf)
ΔTs_long = vcat(-1000:25:-25, 25:25:1000) .* ms
ax3 = kernel_axis!(fig, (2, 1), stdp_sym, ΔTs_long; title = "Symmetric Structured STDP")

stdp_anti = SNN.STDPAntiSymmetric(
    αpre = 0.0f0,
    αpost = 0,
    A_x = 1,
    A_y = 1,
    Wmax = Inf,
    Wmin = -Inf,
)
ax4 = kernel_axis!(fig, (2, 2), stdp_anti, ΔTs_long; title = "Anti Symmetric Structured STDP")
fig

## Triplet and weight-dependent rules (new in SNNModels 1.9). The pair term of the triplet rule is
## the classical window; the weight-dependent rule has soft bounds, so its kernel depends on w (with μ = 0 the amplitudes are η*(Wmax-Wmin) = 1e-2 for LTP and 1.05e-2 for LTD).
stdp_triplet = SNN.STDPTriplet(Wmax = Inf, Wmin = -Inf)    # Pfister & Gerstner 2006, Table 4
stdp_wd = SNN.STDPWeightDependent(η = 3.3e-4, α = 1.05, μ_plus = 0.0, μ_minus = 0.0,
                                  τpre = 16.8ms, τpost = 33.7ms, Wmax = 30pF, Wmin = 0pF)
fig2 = Figure(size = (900, 350))
kernel_axis!(fig2, (1, 1), stdp_triplet, ΔTs; title = "Triplet STDP (pairs only)")
kernel_axis!(fig2, (1, 2), stdp_wd, ΔTs; title = "Weight-dependent STDP (mu = 0, w = 1)")
fig2
