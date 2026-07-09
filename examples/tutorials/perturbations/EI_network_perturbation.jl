##
# Perturbation tutorial: E/I network with current injection to E or I population
#
# Workflow:
#   1. Warm up a balanced E/I network (no recording)
#   2. Set up voltage monitoring
#   3. Run two perturbation tests from the current state — one injecting 1 nA
#      into the inhibitory population, one into the excitatory population
#   4. Advance the baseline for the same 500 ms window
#   5. Recover spliced traces with perturbation_record and compare
#
# Because make_copy is a shallow copy (state arrays shared), each perturbation
# uses from_state = a fresh deepcopy of the model at the checkpoint so that
# setting I on the copy does not alias back to the archive.

using SpikingNeuralNetworks
SNN.@load_units
using Statistics
using CairoMakie

# ── Network ──────────────────────────────────────────────────────────────────

function initialize()
    E = SNN.IF(; N = 2000, synapse=SNN.SingleExpSynapse())
    I = SNN.IF(; N = 500, synapse=SNN.SingleExpSynapse())
    EE = SNN.SpikingSynapse(E, E, :ge; conn = (μ = 2.0, p = 0.2))
    EI = SNN.SpikingSynapse(E, I, :ge; conn = (μ = 10.0, p = 0.2))
    IE = SNN.SpikingSynapse(I, E, :gi; conn = (μ = 10.0, p = 0.2))
    II = SNN.SpikingSynapse(I, I, :gi; conn = (μ = 10.0, p = 0.2))
    inputs = SNN.Poisson(; N = 200, param = SNN.PoissonHomoParameter(rate = 10.5Hz))
    ProjE = SNN.SpikingSynapse(inputs, E, :ge; conn = (μ = 5, p = 0.2))
    P = (; E, I, inputs)
    C = (; EE, EI, IE, II, ProjE)
    monitor!(E, [:v, :fire], sr=0.1kHz)
    monitor!(I, [:v, :fire], sr=0.1kHz)
    return SNN.compose(; P..., C..., silent = true)
end

# ── Warm-up (no recording) ────────────────────────────────────────────────────

model = initialize()
SNN.sim!(model, 1second)            # settle into balanced activity

# ── Monitoring ────────────────────────────────────────────────────────────────

# ── Checkpoint helper ─────────────────────────────────────────────────────────
#
# make_copy is shallow (state arrays aliased), so we deepcopy, clear records,
# re-apply monitoring, then apply the condition for full state independence.

fs = SNN.modelcopy(model)

function make_checkpoint(model, condition!)
    fs = SNN.modelcopy(model)
    condition!(fs)
    return fs
end

# ── Perturbation tests ────────────────────────────────────────────────────────
#
# Both perturbations start from the same checkpoint (t = 1 s) and run for
# 500 ms, so the perturbation window matches the upcoming baseline window.

duration = 1500ms


fs_I = SNN.modelcopy(model)  # checkpoint for I perturbation
fs_E = SNN.modelcopy(model)  # checkpoint for E perturbation


# perturbation_test(model, duration; from_state = fs_I, add_records = "I_input", trigger! = m -> (m.pop.I.I .= 0.4nA))
# perturbation_test(model, duration; from_state = fs_E, add_records = "E_input", trigger! = m -> (m.pop.E.I .= 0.4nA))

SNN.sim!(model, duration)           # record baseline over the same 500 ms window

size(fs_E.pop.E.records[:v])

fs_E

# ── Baseline ──────────────────────────────────────────────────────────────────

SNN.sim!(model, duration)           # record baseline over the same 500 ms window

#
fs_I = make_checkpoint(model, m -> (m.pop.I.I .= 0.2nA))   # inject into I
fs_E = make_checkpoint(model, m -> (m.pop.E.I .= 0.2nA))   # inject into E

perturbation_test(model, duration; from_state = fs_I, add_records = "I_input")
perturbation_test(model, duration; from_state = fs_E, add_records = "E_input")

# ── Baseline ──────────────────────────────────────────────────────────────────

SNN.sim!(model, duration)           # record baseline over the same 500 ms window


# ── Retrieve traces ───────────────────────────────────────────────────────────

# Infer the baseline interval from the stored recording bounds.
t0 = model.pop.E.records[:start_time][:v]
t1 = model.pop.E.records[:end_time][:v]
interval = t0:0.5f0:t1             # 0.5 ms step = 2 kHz

v_E_base, r = perturbation_record(model.pop.E, :v, "none_existing", interval)
v_E_Ipert,  ri = perturbation_record(model.pop.E, :v, "I_input",     interval)
v_E_Epert,  _ = perturbation_record(model.pop.E, :v, "E_input",     interval)


v_I_base, _ = perturbation_record(model.pop.I, :v, "none_existing", interval)
v_I_Ipert, _ = perturbation_record(model.pop.I, :v, "I_input",      interval)
v_I_Epert, _ = perturbation_record(model.pop.I, :v, "E_input",      interval)

r_ms = collect(r)                  # time axis in ms

# ── Plot ──────────────────────────────────────────────────────────────────────

mean_v(mat) = vec(mean(mat; dims = 1))

fig = Figure(size = (900, 600))

axE = Axis(fig[1, 1];
    title   = "Excitatory population — mean Vm",
    xlabel  = "Time (ms)",
    ylabel  = "V (mV)",
    xgridvisible = false,
    ygridvisible = false,
)
lines!(axE, r_ms, mean_v(v_E_base);  label = "baseline", color = :black, linewidth = 2)
lines!(axE, r_ms, mean_v(v_E_Ipert); label = "1 nA → I", color = :blue,  linewidth = 2)
lines!(axE, r_ms, mean_v(v_E_Epert); label = "1 nA → E", color = :red,   linewidth = 2)
axislegend(axE; position = :rt, framevisible = false)

axI = Axis(fig[2, 1];
    title   = "Inhibitory population — mean Vm",
    xlabel  = "Time (ms)",
    ylabel  = "V (mV)",
    xgridvisible = false,
    ygridvisible = false,
)
lines!(axI, r_ms, mean_v(v_I_base);  label = "baseline", color = :black, linewidth = 2)
lines!(axI, r_ms, mean_v(v_I_Ipert); label = "1 nA → I", color = :blue,  linewidth = 2)
lines!(axI, r_ms, mean_v(v_I_Epert); label = "1 nA → E", color = :red,   linewidth = 2)
axislegend(axI; position = :rt, framevisible = false)

fig

##
# Firing-rate summary
SNN.raster(model.pop.I)
##
ss_I, _ = perturbation_record(model.pop.E, :fire, "I_input",      interval)
ss_E, _ = perturbation_record(model.pop.E, :fire, "E_input",      interval)
fr_base,  r_fr, = SNN.firing_rate(model.pop.E, interval = t0:10ms:t1, pop_average = true)
fr_Ipert, r_fr,  = SNN.firing_rate(ss_I,  interval = t0:10ms:t1, pop_average = true)
fr_Epert, r_fr,  = SNN.firing_rate(ss_E,  interval = t0:10ms:t1, pop_average = true)

f = Figure(size = (900, 400))
ax = Axis(f[1, 1]; 
    title   = "Firing rates (mean over 500 ms)",
    xlabel  = "Time (ms)",
    ylabel  = "Firing rate (Hz)",
    xgridvisible = false,
    ygridvisible = false,
)
lines!(ax, r_fr, fr_base;  label = "baseline", color = :black, linewidth = 2)
lines!(ax, r_fr, fr_Ipert; label = "1 nA → I", color = :blue,  linewidth = 2)
lines!(ax, r_fr, fr_Epert; label = "1 nA → E", color = :red,   linewidth = 2)
axislegend(ax; position = :rt, framevisible = false)
f   
##

@info "Mean firing rates over 500 ms (E / I):"
@info "  Baseline:  E = $(round(mean(fr_base[1]),  digits=1)) Hz,  I = $(round(mean(fr_base[2]),  digits=1)) Hz"
@info "  1 nA → I:  E = $(round(mean(fr_Ipert[1]), digits=1)) Hz,  I = $(round(mean(fr_Ipert[2]), digits=1)) Hz"
@info "  1 nA → E:  E = $(round(mean(fr_Epert[1]), digits=1)) Hz,  I = $(round(mean(fr_Epert[2]), digits=1)) Hz"
