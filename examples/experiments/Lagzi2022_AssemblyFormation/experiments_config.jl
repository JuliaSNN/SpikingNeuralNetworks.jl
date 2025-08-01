response_experiments = let
    base_config = (
        NSSTs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        load_path = "baseline",
        ## Response test parameters
        delay = 1s,
        repetitions = 40,
        warmup = 20s,
        target_pop = :E1,
        rec_interval = 0:3s,
        train = false,
        input_strength = 200,
        force = false,
        ## Network parameters
        change_weights = true,
        w_same = 1.0,
        w_cross = 0.5,
        noise_strength = 0.8,
    );
    (
        # "1_HetEE",
        hetEE = (; base_config..., name = "1_HetEE", change_weights = false),
        sameEE = (; base_config..., name = "2_SameEE"),
        low_noise = (; base_config..., name = "3_LowNoise", noise_strength = 0.4),
        low_input = (; base_config..., name = "4_LowInput", input_strength = 100),
        low_input2 = (; base_config..., name = "5_LowInput", input_strength = 50),
        stp = (; base_config..., name = "6_STP", plasticity = SNN.STPParameter()),
        highEE = (; base_config..., name = "7_highEE", w_same = 3.0, w_cross = 1.0),
    )
end

response_experiments_names = keys(response_experiments)

running_exps = [:hetEE, :sameEE, :low_noise, :low_input, :low_input2, :stp, :highEE]

@unpack rates, interval = load(datadir("zeus", "ExpData", "ACrates.jld2")) |> dict2ntuple
sound = mean(rates)

function update_model_parameters!(model; exp_config, config)

    no_ext_input = filter_items(model.stim, condition = x->!occursin("ExtSignal", x.name))
    model = merge_models(model.pop, no_ext_input, model.syn, silent = true)

    # Set external noise
    # @unpack ext_rate = config
    ext_rate = 2kHz
    @unpack noise_strength = exp_config
    # model.stim.inh_noise.param.rate .= ext_rate * noise_strength
    model.stim.exc_noise1.param.rate .= ext_rate * noise_strength
    model.stim.exc_noise2.param.rate .= ext_rate * noise_strength

    if exp_config.change_weights
        @unpack w_same, w_cross = exp_config
        model.syn.E1_to_E1.W .*= w_same/mean(model.syn.E1_to_E1.W)
        model.syn.E2_to_E2.W .*= w_same/mean(model.syn.E1_to_E1.W)
        model.syn.E1_to_E2.W .*= w_cross/mean(model.syn.E1_to_E1.W)
        model.syn.E2_to_E1.W .*= w_cross/mean(model.syn.E1_to_E1.W)
    end
    if haskey(exp_config, :plasticity)
        @unpack plasticity = exp_config
        model.syn.E1_to_E1.param = plasticity
        model.syn.E2_to_E2.param = plasticity
        model.syn.E1_to_E2.param = plasticity
        model.syn.E2_to_E1.param = plasticity
    end
    return model
end

# Remove external signal

# Set model, stimulus and exp parameters
