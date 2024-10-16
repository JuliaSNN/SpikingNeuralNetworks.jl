
@snn_kw struct STDPParameter{FT = Float32} <: SpikingSynapseParameter
    τpre::FT = 20ms
    τpost::FT = 20ms
    Wmax::FT = 0.01
    ΔApre::FT = 0.01 * Wmax
    ΔApost::FT = -ΔApre * τpre / τpost * 1.05
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

function get_variables(param::STDPParameter, Npre, Npost)
    return STDPVariables(Npre = Npre, Npost = Npost)
end

## It's broken   !!

function plasticity!(c::AbstractSparseSynapse, param::STDPParameter, dt::Float32, T::Time)
    plasticity!(c, param, c.plasticity, dt, T)
end

function plasticity!(
    c::AbstractSparseSynapse,
    param::STDPParameter,
    plasticity::STDPVariables,
    dt::Float32,
    T::Time,
)
    @unpack rowptr, colptr, I, J, index, W, fireI, fireJ, g = c
    @unpack τpre, τpost, Wmax, ΔApre, ΔApost = plasticity

    @inbounds for j = 1:(length(colptr)-1)
        if fireJ[j]
            for s = colptr[j]:(colptr[j+1]-1)
                Apre[s] *= exp32(-(t - tpre[s]) / τpre)
                Apost[s] *= exp32(-(t - tpost[s]) / τpost)
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
                Apre[s] *= exp32(-(t - tpre[s]) / τpre)
                Apost[s] *= exp32(-(t - tpost[s]) / τpost)
                Apost[s] += ΔApost
                tpost[s] = t
                W[s] = clamp(W[s] + Apre[s], 0.0f0, Wmax)
            end
        end
    end
end

export STDPParameter, STDPVariables, get_variables, plasticity!
