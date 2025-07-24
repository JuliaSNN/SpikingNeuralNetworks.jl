
@snn_kw struct STDPParameter{FT = Float32} <: SpikingSynapseParameter
    τpre::FT = 20ms
    τpost::FT = 20ms
    Wmax::FT = 0.01
    ΔApre::FT = 0.01 * Wmax
    ΔApost::FT = -ΔApre * τpre / τpost * 1.05
    active::Vector{Bool} = [true]
end

@snn_kw struct STDPVariables{VFT = Vector{Float32},IT = Int} <: PlasticityVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    tpre::VFT = zeros(Npre) # presynaptic spiking time
    tpost::VFT = zeros(Npost) # postsynaptic spiking time
    Apre::VFT = zeros(Npre) # presynaptic trace
    Apost::VFT = zeros(Npost) # postsynaptic trace
end

function plasticityvariables(param::STDPParameter, Npre, Npost)
    return STDPVariables(Npre = Npre, Npost = Npost)
end

## It's broken   !!

function plasticity!(c::AbstractSparseSynapse, param::STDPParameter, dt::Float32)
    @unpack active = param
    !active[1] && return
    plasticity!(c, param, c.plasticity, dt)
end

function plasticity!(
    c::AbstractSparseSynapse,
    param::STDPParameter,
    plasticity::STDPVariables,
    dt::Float32,
)
    @unpack rowptr, colptr, I, J, index, W, fireI, fireJ, g = c
    @unpack τpre, τpost, Wmax, ΔApre, ΔApost = plasticity

    @inbounds for j = 1:(length(colptr)-1)
        if fireJ[j]
            for s = colptr[j]:(colptr[j+1]-1)
                Apre[s] *= exp64(-(t - tpre[s]) / τpre)
                Apost[s] *= exp64(-(t - tpost[s]) / τpost)
                Apre[s] += ΔApre
                tpre[s] = t
                W[s] = clamp(W[s] + Apost[s], 0.0f0, Wmax)
            end
        end
    end
    @inbounds for i = 1:(length(rowptr)-1)
        if fireI[i]
            for st = rowptr[i]:(rowptr[i+1]-1)
                s = index[st]
                Apre[s] *= exp64(-(t - tpre[s]) / τpre)
                Apost[s] *= exp64(-(t - tpost[s]) / τpost)
                Apost[s] += ΔApost
                tpost[s] = t
                W[s] = clamp(W[s] + Apre[s], 0.0f0, Wmax)
            end
        end
    end
end


# using SNNUtils
# using SNNPlots

# triplet = SNNUtils.pfister_visualcortex()
# o1stdp = 0.0
# o2stdp = 0.0
# r1stdp = 0.0
# r2stdp = 0.0

# pre_rate = 20
# post_rate = 30
# simtime = 1000
# dt = 0.1f0
# pre_spikes = SNNUtils.PoissonInput(pre_rate, simtime, dt)[1, :]
# post_spikes = SNNUtils.PoissonInput(post_rate, simtime, dt)[1, :]

# for tt = 1:round(Int, simtime / dt)
#     # ## Duplet traces update before learning rule
#     pre_spiked = pre_spikes[tt]
#     post_spiked = post_spikes[tt]

#     post_spiked && (o1stdp += 1.0)
#     pre_spiked && (pre_spiked += 1.0)

#     if exc_prespikes[syn]
#         W -= o1stdp * (triplet.A⁻₂ + triplet.A⁻₃ * r2stdp[syn])
#     end

#     if post_spiked
#         W += r1stdp[syn] * (triplet.A⁺₂ + triplet.A⁺₃ * o2stdp)
#     end

#     post_spiked && (o2stdp += 1.0)
#     pre_spiked && (r2stdp += 1.0)

#     r2stdp .*= exp(-dt / triplet.τˣ)
#     o2stdp *= exp(-dt / triplet.τʸ)
#     r1stdp .*= exp(-dt / triplet.τ⁺)
#     o1stdp *= exp(-dt / triplet.τ⁻)
# end
# # AMPAsynapses[findall(AMPAsynapses .> max_efficacy)] .= max_efficacy
# # AMPAsynapses[findall(AMPAsynapses .< 0.)] .=0

export STDPParameter, STDPVariables, plasticityvariables, plasticity!
