@snn_kw struct MarkramSTPParameter{FT = Float32} <: STPParameter
    τD::FT = 200ms # τx
    τF::FT = 1500ms # τu
    U::FT = 0.2
    Wmax::FT = 1.0pF
    Wmin::FT = 0.0pF
end

@snn_kw struct MarkramSTPVariables{VFT = Vector{Float32},IT = Int} <: STPVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    u::VFT = zeros(Npost) # presynaptic spiking time
    x::VFT = ones(Npost) # postsynaptic spiking time
    _ρ::VFT = ones(Npost) # postsynaptic spiking time
    active::Vector{Bool} = [true]
end

plasticityvariables(param::MarkramSTPParameter, Npre, Npost) = MarkramSTPVariables(Npre = Npre, Npost = Npost)


function plasticity!(
    c::PT,
    param::MarkramSTPParameter,
    plasticity::MarkramSTPVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, v_post, fireJ, g, ρ, index = c
    @unpack u, x, _ρ = plasticity
    @unpack U, τF, τD, Wmax, Wmin = param

    @simd for j in eachindex(fireJ) # Iterate over all columns, j: presynaptic neuron
        if fireJ[j]
            u[j] += U * (1 - u[j])
            x[j] += (-u[j] * x[j])
        end
    end

    # update pre-synaptic spike trace
    @turbo for j in eachindex(fireJ) # Iterate over all columns, j: presynaptic neuron
        @fastmath u[j] += dt * (U - u[j]) / τF # facilitation
        @fastmath x[j] += dt * (1 - x[j]) / τD # depression
        @fastmath _ρ[j] = u[j] * x[j]
    end

    Threads.@threads :static for j in eachindex(fireJ) # Iterate over postsynaptic neurons
        @inbounds @simd for s = colptr[j]:(colptr[j+1]-1)
            ρ[s] = _ρ[j]
        end
    end
end

export MarkramSTPParameter, MarkramSTPVariables, plasticityvariables, plasticity!
