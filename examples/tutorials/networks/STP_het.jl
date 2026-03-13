using Random 
using CSV
using DataFrames, DataFramesMeta
using CairoMakie

data = CSV.read(joinpath(ASSET_PATH, "dsxc_mouse_fit_results.csv"), DataFrame)
@rtransform! data :post_cell_class= $(Symbol("post_cell.cell_class"))
@rsubset! data :fit_tau_fac .< 10
@rsubset! data :fit_tau_rec .< 10
@rtransform! data :fit_tau_fac = :fit_tau_fac .* 1000
@rtransform! data :fit_tau_rec = :fit_tau_rec .* 1000
@rsubset! data :fit_mse < quantile(data.fit_mse, 0.75)
@rsubset! data :fit_U .> 0.1 && :fit_U .< 0.9

dropmissing!(data, :post_cell_class)
inh_inh_df = @rsubset data String(:synapse_type) == "in" && String(:post_cell_class) == "in"
exc_inh_df = @rsubset data String(:synapse_type) == "ex" && String(:post_cell_class) == "in"
exc_exc_df = @rsubset data String(:synapse_type) == "ex" && String(:post_cell_class) == "ex"
inh_exc_df = @rsubset data String(:synapse_type) == "in" && String(:post_cell_class) == "ex"


function sample_stp_params(df, N)
    d = map(1:N) do n
        [df[!,x][rand(1:nrow(df))] for x in  [:fit_tau_rec, :fit_tau_fac, :fit_U, :fit_w]]
    end  |> x->reduce(hcat, x)
    return (;τF = d[2,:], τD = d[1,:], U = d[3,:])
end


## AdEx neuron with fixed external current connections with multiple receptors
E_uni = SNN.AdExParameter(; El = -50mV)
E_het = SNN.heterogeneous(E_uni, 3200; τm = Distributions.Normal(10.0f0, 2.0f0), b = Distributions.Normal(60.0f0, 4.0f0))

E = SNN.Population(E_het, synapse = SNN.DoubleExpSynapse(); N = 3200, name = "Excitatory")

I = SNN.Population(
    SNN.IFParameter(),
    synapse = SNN.SingleExpSynapse();
    N = 800,
    name = "Inhibitory",
    spike = SNN.PostSpike(),
)

EE = SNN.SpikingSynapse(E, E, :he; conn = (μ = 1, p = 0.02), 
        STPParam = SNN.MarkramSTPParameterHet(;sample_stp_params(exc_exc_df, E.N)...))
EI = SNN.SpikingSynapse(E, I, :ge; conn = (μ = 30, p = 0.02),
        STPParam = SNN.MarkramSTPParameterHet(;sample_stp_params(exc_inh_df, E.N)...))
IE = SNN.SpikingSynapse(I, E, :hi; conn = (μ = 50, p = 0.02),
        STPParam = SNN.MarkramSTPParameterHet(;sample_stp_params(inh_exc_df, I.N)...))
II = SNN.SpikingSynapse(I, I, :gi; conn = (μ = 10, p = 0.02),
        STPParam = SNN.MarkramSTPParameterHet(;sample_stp_params(inh_inh_df, I.N)...))
model = SNN.compose(; E, I, EE, EI, IE, II)

SNN.monitor!(model.pop, [:fire])
SNN.monitor!(EE, [:ρ], sr=10Hz)
SNN.monitor!(EE, [:x, :u], sr=10Hz, variables=:STPVars)
SNN.train!(model = model; duration = 5second, pbar = true)
## Get the recorded variables and plot the effective synaptic efficacy for the EE synapse
ρ, r =  SNN.record(EE, :ρ, range=true) 
u, r =  SNN.record(EE, :STPVars_u, range=true)
x, r =  SNN.record(EE, :STPVars_x, range=true)
jj = EE.colptr[1:end-1]
series(ρ(jj[1:256],r), color=:viridis)
##
SNN.clear_monitor!(model)
SNN.monitor!(model.pop, [:fire])
SNN.reset_time!(model)
SNN.train!(model = model; duration = 5second, pbar = true)
SNN.raster(model.pop, 4s:5s)
##
ssn = SNN.spiketimes(model.pop.E)[1:100]
SNN.STTC(ssn, 50ms, 0:1s ) |> mean

