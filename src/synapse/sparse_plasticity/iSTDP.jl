abstract type iSTDPParameter <: STDPParameter end

@snn_kw struct iSTDPRate{FT = Float32} <: iSTDPParameter
    η::FT = 0.01pA
    r::FT = 3Hz
    τy::FT = 50ms
    Wmax::FT = 243pF
    Wmin::FT = 0.01pF
end

@snn_kw struct iSTDPTime{FT = Float32} <: iSTDPParameter
    η::FT = 0.01pA
    τy::FT = 50ms
    Wmax::FT = 243pF
    Wmin::FT = 0.01pF
end

@snn_kw mutable struct iSTDPPotential{FT = Float32} <: iSTDPParameter
    η::FT = 0.001pA
    v0::FT = -50mV
    τy::FT = 200ms
    Wmax::FT = 243pF
    Wmin::FT = 0.01pF
end


@snn_kw struct iSTDPVariables{VFT = Vector{Float32},IT = Int} <: PlasticityVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    tpost::VFT = zeros(Npost) # postsynaptic spiking time 
    tpre::VFT = zeros(Npre) # presynaptic spiking time
    active::Vector{Bool} = [true]
end

function plasticityvariables(param::T, Npre, Npost) where {T<:iSTDPParameter}
    return iSTDPVariables(Npre = Npre, Npost = Npost)
end

"""
    plasticity!(c::AbstractSparseSynapse, param::iSTDPRate, dt::Float32)

Performs the synaptic plasticity calculation based on the inihibitory spike-timing dependent plasticity (iSTDP) model from Vogels (2011). 
The function updates synaptic weights `W` of each synapse in the network according to the firing status of pre and post-synaptic neurons.
This is an in-place operation that modifies the input `AbstractSparseSynapse` object `c`.

# Arguments
- `c::AbstractSparseSynapse`: The current spiking synapse object which contains data structures to represent the synapse network.
- `param::iSTDPRate`: Parameters needed for the iSTDP model, including learning rate `η`, target rate `r`, STDP time constant `τy`, maximal and minimal synaptic weight (`Wmax` and `Wmin`).
- `dt::Float32`: The time step for the numerical integration.

# Algorithm
- For each pre-synaptic neuron, if it fires, it reduces the synaptic weight by an amount proportional to the difference between the target rate and the actual rate and increases the inhibitory term, 
  otherwise the inhibitory term decays exponentially over time with a time constant `τy`.
- For each post-synaptic neuron, if it fires, it increases the synaptic weight by an amount proportional to the pre-synaptic trace and increases the excitatory term,
  otherwise the excitatory term decays exponentially over time with a time constant `τy`.
- The synaptic weights are bounded by `Wmin` and `Wmax`.
"""
function plasticity!(
    c::AbstractSparseSynapse,
    param::iSTDPRate,
    plasticity::iSTDPVariables,
    dt::Float32,
    T::Time,
)
    @unpack rowptr, colptr, index, I, J, W, fireI, fireJ, g = c
    @unpack η, r, τy, Wmax, Wmin = param
    @unpack tpre, tpost = plasticity
    # @inbounds 
    # if pre-synaptic inhibitory neuron fires, it aims to modify the synaptic weight to achieve the target post synaptic rate
    @fastmath begin
        @inbounds for j in eachindex(fireJ) # presynaptic indices j
            tpre[j] += dt * (-tpre[j]) / τy
            if fireJ[j] # presynaptic neuron
                tpre[j] += 1
                @turbo for st = colptr[j]:(colptr[j+1]-1)
                    W[st] = clamp(W[st] + η * (tpost[I[st]] - 2 * r * τy), Wmin, Wmax)
                end
            end
        end
        # if post-synaptic excitatory neuron fires
        # @inbounds 
        @inbounds for i in eachindex(fireI) # postsynaptic indices i
            tpost[i] += dt * (-tpost[i]) / τy
            if fireI[i] # postsynaptic neuron
                tpost[i] += 1
                @turbo for st = rowptr[i]:(rowptr[i+1]-1) ## 
                    st = index[st]
                    W[st] = clamp(W[st] + η * tpre[J[st]], Wmin, Wmax)
                end
            end
        end
    end
end

##
function plasticity!(
    c::AbstractSparseSynapse,
    param::iSTDPTime,
    plasticity::iSTDPVariables,
    dt::Float32,
    T::Time,
)
    @unpack rowptr, colptr, index, I, J, W, fireI, fireJ, g = c
    @unpack η, τy, Wmax, Wmin = param
    @unpack tpre, tpost = plasticity
    # @inbounds 
    # if pre-synaptic inhibitory neuron fires, it aims to modify the synaptic weight to achieve the target post synaptic rate
    @fastmath begin
        @inbounds for j in eachindex(fireJ) # presynaptic indices j
            tpre[j] += dt * (-tpre[j]) / τy
            if fireJ[j] # presynaptic neuron
                tpre[j] += 1
                @turbo for st = colptr[j]:(colptr[j+1]-1)
                    W[st] = clamp(W[st] + η * (tpost[I[st]] - 1 / τy), Wmin, Wmax)
                end
            end
        end
        # if post-synaptic excitatory neuron fires
        @inbounds for i in eachindex(fireI) # postsynaptic indices i
            tpost[i] += dt * (-tpost[i]) / τy
            if fireI[i] # postsynaptic neuron
                tpost[i] += 1
                @turbo for st = rowptr[i]:(rowptr[i+1]-1) ## 
                    st = index[st]
                    W[st] = clamp(W[st] + η * (tpre[J[st]]), Wmin, Wmax)
                end
            end
        end
    end
end


"""
    plasticity!(c::AbstractSparseSynapse, param::iSTDPRate, dt::Float32)

Performs the synaptic plasticity calculation based on the inihibitory spike-timing dependent plasticity (iSTDP) model from Vogels (2011) adapted to control the membrane potential. 
The function updates synaptic weights `W` of each synapse in the network according to the firing status of pre and post-synaptic neurons.
This is an in-place operation that modifies the input `AbstractSparseSynapse` object `c`.

# Arguments
- `c::AbstractSparseSynapse`: The current spiking synapse object which contains data structures to represent the synapse network.
- `param::iSTDPParameterVoltage`: Parameters needed for the iSTDP model, including learning rate `η`, target membrane potenital `v0`, STDP time constant `τy`, maximal and minimal synaptic weight (`Wmax` and `Wmin`).
- `dt::Float32`: The time step for the numerical integration.

# Algorithm
- For each pre-synaptic neuron, if it fires, it reduces the synaptic weight by an amount proportional to the difference between the target membrane potential and the actual one. 
- For each pre-synaptic neuron, if it fires, increases the inhibitory term, otherwise the inhibitory term decays exponentially over time with a time constant `τy`.
- For each post-synaptic neuron, if it fires, it increases the synaptic weight by an amount proportional to the pre-synaptic trace and increases the excitatory term, otherwise the excitatory term decays exponentially over time with a time constant `τy`.
- The synaptic weights are bounded by `Wmin` and `Wmax`.
"""
function plasticity!(
    c::AbstractSparseSynapse,
    param::iSTDPPotential,
    plasticity::iSTDPVariables,
    dt::Float32,
    T::Time,
)
    @unpack rowptr, colptr, index, I, J, W, v_post, fireI, fireJ, g = c
    @unpack η, v0, τy, Wmax, Wmin = param
    @unpack tpre, tpost = plasticity

    # @inbounds 
    # if pre-synaptic inhibitory neuron fires
    @fastmath @inbounds for j in eachindex(fireJ) # presynaptic indices j
        tpre[j] += dt * (-tpre[j]) / τy
        if fireJ[j] # presynaptic neuron
            tpre[j] += 1
            for st = colptr[j]:(colptr[j+1]-1)
                W[st] = clamp(W[st] + η * (tpost[I[st]] - v0), Wmin, Wmax)
            end
        end
    end
    # if post-synaptic excitatory neuron fires
    @fastmath @inbounds for i in eachindex(fireI) # postsynaptic indices i
        # trace of the membrane potential
        tpost[i] += dt * -(tpost[i] - v_post[i]) / τy
        if fireI[i] # postsynaptic neuron
            for st = rowptr[i]:(rowptr[i+1]-1) ## 
                st = index[st]
                W[st] = clamp(W[st] + η * tpre[J[st]], Wmin, Wmax)
            end
        end
    end
end

export iSTDPRate,
    iSTDPTime,
    iSTDPPotential,
    iSTDPVariables,
    plasticityvariables,
    plasticity!
