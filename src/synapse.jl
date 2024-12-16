@snn_kw struct Receptor{T = Float32}
    E_rev::T = 0.0
    τr::T = -1.0f0
    τd::T = -1.0f0
    g0::T = 0.0f0
    gsyn::T = g0 > 0 ? g0 * norm_synapse(τr, τd) : 0.0f0
    α::T = α_synapse(τr, τd)
    τr⁻::T = 1 / τr > 0 ? 1 / τr : 0.0f0
    τd⁻::T = 1 / τd > 0 ? 1 / τd : 0.0f0
    nmda::T = 0.0f0
end

ReceptorVoltage = Receptor
SynapseArray = Vector{Receptor{Float32}}

struct Synapse{T<:Receptor}
    AMPA::T
    NMDA::T
    GABAa::T
    GABAb::T
end

struct Glutamatergic{T<:Receptor}
    AMPA::T
    NMDA::T
end

struct GABAergic{T<:Receptor}
    GABAa::T
    GABAb::T
end

function Synapse(glu::Glutamatergic, gaba::GABAergic)
    return Synapse(glu.AMPA, glu.NMDA, gaba.GABAa, gaba.GABAb)
end

export Receptor,
    Synapse, ReceptorVoltage, GABAergic, Glutamatergic, SynapseArray, NMDAVoltageDependency

function norm_synapse(synapse::Receptor)
    norm_synapse(synapse.τr, synapse.τd)
end


function norm_synapse(τr, τd)
    p = [1, τr, τd]
    t_p = p[2] * p[3] / (p[3] - p[2]) * log(p[3] / p[2])
    return 1 / (-exp(-t_p / p[2]) + exp(-t_p / p[3]))
end

# α is the factor that has to be placed in-front of the differential equation as such the analytical integration corresponds to the double exponential function. Further details are discussed in the Julia notebook about synapses
function α_synapse(τr, τd)
    return (τd - τr) / (τd * τr)
end

function synapsearray(syn::Synapse, indices::Vector = [])::SynapseArray
    container = SynapseArray()
    names = isempty(indices) ? fieldnames(Synapse) : fieldnames(Synapse)[indices]
    for name in names
        receptor = getfield(syn, name)
        if !(receptor.τr  < 0)
            push!(container, receptor)
        end
    end
    return container
end

function synapsearray(syn::SynapseArray)::SynapseArray
    return syn
end


Mg_mM = 1.0f0
nmda_b = 3.36   # voltage dependence of nmda channels
# nmda_k   = -0.062     #(1/V) source: http://dx.doi.org/10.1016/j.neucom.2011.04.018)
nmda_k = -0.077     #Eyal 2018

@snn_kw struct NMDAVoltageDependency{T<:Float32}
    b::T = nmda_b
    k::T = nmda_k
    mg::T = Mg_mM
end


EyalNMDA = NMDAVoltageDependency(mg = Mg_mM, b = nmda_b, k = nmda_k)

export norm_synapse , EyalNMDA
