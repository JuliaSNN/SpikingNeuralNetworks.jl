function network(; local_config, type = :pv1)
    # Number of neurons in the network
    # Create dendrites for each neuron
    @unpack NE, NI, NSST = local_config
    @unpack adex_param, pv_param, sst_param = local_config
    @unpack EI, EE, II, IE, stdp_exc, stdp_pv, stdp_sst = local_config
    @unpack E_noise, I_noise = local_config
    @unpack signal_param = local_config

    E1 = SNN.AdEx(N = NE, param = adex_param, name = "Exc1")
    E2 = SNN.AdEx(N = NE, param = adex_param, name = "Exc2")
    pop = nothing
    noise = nothing
    if type == :pv1
        NPV = NI
        PV = SNN.IF(; N = NPV, param = pv_param, name = "PV")
        pop = (@symdict E1 E2 PV)
        noise = (
            exc_noise1 = SNN.PoissonStimulus(
                E1,
                :he,
                μ = 2mV,
                param = E_noise,
                neurons = :ALL,
            ),
            exc_noise2 = SNN.PoissonStimulus(
                E2,
                :he,
                μ = 2mV,
                param = E_noise,
                neurons = :ALL,
            ),
            inh_noise = SNN.PoissonStimulus(PV, :he, param = I_noise, neurons = :ALL),
        )
    end
    if type == :pv2
        NPV = NI ÷ 2
        PV1 = SNN.IF(; N = NPV, param = pv_param, name = "PV1")
        PV2 = SNN.IF(; N = NPV, param = pv_param, name = "PV2")
        noise = (
            exc_noise1 = SNN.PoissonStimulus(
                E1,
                :he,
                μ = 2mV,
                param = E_noise,
                neurons = :ALL,
            ),
            exc_noise2 = SNN.PoissonStimulus(
                E2,
                :he,
                μ = 2mV,
                param = E_noise,
                neurons = :ALL,
            ),
            inh_noise1 = SNN.PoissonStimulus(PV1, :he, param = I_noise, neurons = :ALL),
            inh_noise2 = SNN.PoissonStimulus(PV2, :he, param = I_noise, neurons = :ALL),
        )
        pop = (@symdict E1 E2 PV1 PV2)
    end
    if type == :sst
        NPV = (NI - 2NSST)
        PV = SNN.IF(; N = NPV, param = pv_param, name = "PV")
        SST1 = SNN.IF(; N = NSST, param = sst_param, name = "SST1")
        SST2 = SNN.IF(; N = NSST, param = sst_param, name = "SST2")
        pop = (@symdict E1 E2 PV SST1 SST2)
        noise = (
            exc_noise1 = SNN.PoissonStimulus(
                E1,
                :he,
                μ = 2mV,
                param = E_noise,
                neurons = :ALL,
            ),
            exc_noise2 = SNN.PoissonStimulus(
                E2,
                :he,
                μ = 2mV,
                param = E_noise,
                neurons = :ALL,
            ),
            inh_noise = SNN.PoissonStimulus(PV, :he, param = I_noise, neurons = :ALL),
        )
    end
    pop = dict2ntuple(pop)
    SNN.monitor(pop, [:fire])
    # SNN.monitor(pop, [:v], sr=50Hz)


    variable_stim1 = PoissonStimulusVariable(
        variables = copy(signal_param),
        rate = SNN.OrnsteinUhlenbeckProcess,
    )
    variable_stim2 = PoissonStimulusVariable(
        variables = copy(signal_param),
        rate = SNN.OrnsteinUhlenbeckProcess,
    )

    signal = (
        signal_E1 = SNN.PoissonStimulus(
            E1,
            :he,
            neurons = :ALL,
            μ = 1.0,
            param = variable_stim1,
            name = "ExtSignal_E1",
        ),
        signal_E2 = SNN.PoissonStimulus(
            E2,
            :he,
            neurons = :ALL,
            μ = 1.0,
            param = variable_stim2,
            name = "ExtSignal_E2",
        ),
    )

    synapses = Dict{Symbol,Any}()
    for i in keys(pop)
        for j in keys(pop)
            pre = string(i)
            post = string(j)
            if pre[1] == 'E'
                if post[1] == 'E'
                    synapse = SNN.SpikingSynapse(
                        pop[i],
                        pop[j],
                        :he,
                        p = EE.p,
                        μ = EE.μ,
                        name = "$(pre)_to_$(post)",
                        param = stdp_exc,
                    )
                    SNN.monitor(synapse, [:W], sr = 4Hz)
                    push!(synapses, Symbol("$(pre)_to_$(post)") => synapse)
                elseif post[1] == 'S'
                    if post[end] == pre[end] # cotuned
                        synapse = SNN.SpikingSynapse(
                            pop[i],
                            pop[j],
                            :he,
                            p = 2/3*EI.p,
                            μ = 2*EI.μ,
                            name = "$(pre)_to_$(post)_cotuned",
                        )
                    else # lateral
                        synapse = SNN.SpikingSynapse(
                            pop[i],
                            pop[j],
                            :he,
                            p = 1/3*EI.p,
                            μ = EI.μ,
                            name = "$(pre)_to_$(post)",
                        )
                    end
                    push!(synapses, Symbol("$(pre)_to_$(post)") => synapse)
                elseif post[1] == 'P'
                    if post[end] == pre[end] # cotuned
                        synapse = SNN.SpikingSynapse(
                            pop[i],
                            pop[j],
                            :he,
                            p = 2/3EI.p,
                            μ = 2*EI.μ,
                            name = "$(pre)_to_$(post)_cotuned",
                        )
                    else # lateral
                        synapse = SNN.SpikingSynapse(
                            pop[i],
                            pop[j],
                            :he,
                            p = 1/3*EI.p,
                            μ = EI.μ,
                            name = "$(pre)_to_$(post)",
                        )
                    end
                    push!(synapses, Symbol("$(pre)_to_$(post)") => synapse)
                end
            elseif pre[1] == 'S'
                if post[1] == 'E'
                    synapse = SNN.SpikingSynapse(
                        pop[i],
                        pop[j],
                        :hi,
                        p = IE.p,
                        μ = IE.μ,
                        name = "$(pre)_to_$(post)",
                        param = stdp_sst,
                    )
                    SNN.monitor(synapse, [:W], sr = 4Hz)
                    push!(synapses, Symbol("$(pre)_to_$(post)") => synapse)
                else
                    synapse = SNN.SpikingSynapse(
                        pop[i],
                        pop[j],
                        :hi,
                        p = II.p,
                        μ = II.μ,
                        name = "$(pre)_to_$(post)",
                    )
                    push!(synapses, Symbol("$(pre)_to_$(post)") => synapse)
                end
            elseif pre[1] == 'P'
                if post[1] == 'E'
                    synapse = SNN.SpikingSynapse(
                        pop[i],
                        pop[j],
                        :hi,
                        p = IE.p,
                        μ = IE.μ,
                        name = "$(pre)_to_$(post)",
                        param = stdp_pv,
                    )
                    SNN.monitor(synapse, [:W], sr = 4Hz)
                    push!(synapses, Symbol("$(pre)_to_$(post)") => synapse)
                else
                    synapse = SNN.SpikingSynapse(
                        pop[i],
                        pop[j],
                        :hi,
                        p = II.p,
                        μ = II.μ,
                        name = "$(pre)_to_$(post)",
                    )
                    push!(synapses, Symbol("$(pre)_to_$(post)") => synapse)
                end
            end
        end
    end
    compose(noise, pop, synapses; signal)
end
