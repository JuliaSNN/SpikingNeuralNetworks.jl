@snn_kw struct STPParameter{FT = Float32} <: SpikingSynapseParameter
    τD::FT = 1500ms
    τF::FT = 200ms
    U::FT = 0.2
    Wmax::FT = 1.0pF
    Wmin::FT = 0.0pF
end

@snn_kw struct STPVariables{VFT = Vector{Float32},IT = Int} <: PlasticityVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    u::VFT = zeros(Npost) # presynaptic spiking time
    x::VFT = zeros(Npost) # postsynaptic spiking time
    _ρ::VFT = zeros(Npost) # postsynaptic spiking time
end

function get_variables(param::STPParameter, Npre, Npost)
    return STPVariables(Npre = Npre, Npost = Npost)
end

"""
    plasticity!(c::AbstractSparseSynapse, param::vSTDPParameter, dt::Float32)

Perform update of synapses using plasticity rules based on the Spike Timing Dependent Plasticity (STDP) model.
This function updates pre-synaptic spike traces and post-synaptic membrane traces, and modifies synaptic weights using vSTDP rules.

# Arguments
- `c::AbstractSparseSynapse`: The spiking synapse to be updated.
- `param::vSTDPParameter`: Contains STDP parameters including A_LTD, A_LTP, θ_LTD, θ_LTP, τu, τv, τx, Wmax, Wmin.
    - `A_LTD`: Long Term Depression learning rate.
    - `A_LTP`: Long Term Potentiation learning rate.
    - `θ_LTD`: LTD threshold.
    - `θ_LTP`: LTP threshold.
    - `τu, τv, τx`: Time constants for different variables in STDP.
    - `Wmax, Wmin`: Maximum and minimum synaptic weight.
- `dt::Float32`: Time step for simulation.

In addition to these, the function uses normalization where the operator can be multiplicative or additive as defined by `c.normalize.param.operator`.
The `operator` is applied when updating the synaptic weights. The frequency of normalization is controlled by `τ`, 
where if `τ > 0.0f0` then normalization will occur at intervals approximately equal to `τ`.

After all updates, the synaptic weights are clamped between `Wmin` and `Wmax`.

"""
function plasticity!(c::AbstractSparseSynapse, param::STPParameter, dt::Float32, T::Time)
    plasticity!(c, param, c.plasticity, dt, T)
end

function plasticity!(
    c::AbstractSparseSynapse,
    param::STPParameter,
    plasticity::STPVariables,
    dt::Float32,
    T::Time,
)
    @unpack rowptr, colptr, I, J, index, W, v_post, fireJ, g, ρ,index = c
    @unpack u, x, _ρ = plasticity
    @unpack U, τF, τD, Wmax, Wmin = param

    # update pre-synaptic spike trace
    @turbo for j in eachindex(fireJ) # Iterate over all columns, j: presynaptic neuron
        @fastmath u[j] += dt * (U- u[j])/τF  
        @fastmath x[j] += dt * (1- x[j])/τD 
    end

    @simd for j in eachindex(fireJ) # Iterate over all columns, j: presynaptic neuron
        if fireJ[j]
            u[j] += dt*(U * (1 - x[j]))
            x[j] += dt*(- u[j] * x[j])
            _ρ[j] = u[j] * x[j]
        end
    end

    Threads.@threads for j in eachindex(fireJ) # Iterate over postsynaptic neurons
        @inbounds @simd for s = colptr[j]:(colptr[j+1]-1)
            ρ[s] = _ρ[j]
        end
    end
end

export  STPParameter, STPVariables, get_variables, plasticity!