using SpikingNeuralNetworks
using Distributions
using Plots
SNN.@load_units

E = SNN.MorrisLecar(N = 1)
I = SNN.CurrentStimulus(
    E,
    :I,
    param = SNN.CurrentNoiseParameter(E.N, I_dist = Normal(100pA, 200pA)),
)
SNN.monitor!(E, [:v, :w])

model = SNN.merge_models(; E, I)
E.I .= 100pA
SNN.sim!(model, 1s)

plot(SNN.vecplot(E, :v), SNN.vecplot(E, :w), layout = (2, 1))

##


ws = Float32.(0:0.15:1)
vs = Float32.(-25:8:45)
ds = zeros(2, length(vs), length(ws))
w_nullcline = zeros(length(vs))
v_nullcline = zeros(length(vs))
I = Float32(100pA)
for i in eachindex(vs)
    for j in eachindex(ws)
        ds[1, i, j] = SNNModels.MorrisLecar_dv(vs[i], ws[j], I, E.param)/20
        ds[2, i, j] = SNNModels.MorrisLecar_dw(vs[i], ws[j], E.param)/1

        v_nullcline[i] = SNNModels.MorrisLecar_v_nullcline(vs[i], I, E.param)
        w_nullcline[i] = - SNNModels.MorrisLecar_w_nullcline(vs[i], E.param)
    end
end

quiver(
    repeat(vs, outer = length(ws)),
    repeat(ws, inner = length(vs)),
    quiver = (ds[1, :, :][:], ds[2, :, :][:]),
    c = :black,
    arrow = :filled,
)



plot!(vs, w_nullcline, label = "w nullcline", color = :blue)
plot!(vs, v_nullcline, label = "v nullcline", color = :red)
plot!(legend = :outerright)
