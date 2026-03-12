module SpikingNeuralNetworks

    using SNNModels
    using SNNPlots
    using SNNUtils

    export SNNModels, SNNPlots, SNNUtils

    SNN = SpikingNeuralNetworks
    export SNN

    DOCS_ASSETS_PATH =
        joinpath(dirname(dirname(pathof(SpikingNeuralNetworks))), "docs", "src", "assets")
    export DOCS_ASSETS_PATH

    export SNNPlots, vecplot, raster, vecplot!, raster!, @makie_default, okabe_ito_10

    @makie_default
    @load_units
    
    ## from SNNUtils
    export SNNUtils, SVCtrain

    ## Components from SNNModels
    export AdExParameter, DendNeuronParameter, DoubleExpSynapse,
    ExtendedIFParameter, GABAergic, Glutamatergic, IFParameter, Identity,
    MarkramSTPParameter, MultiplicativeNorm, NMDAVoltageDependency, NoLTP, NoSTP, Poisson,
    PoissonParameter, Population, PostSpike, Receptor, ReceptorSynapse, ReceptorVoltage,
    Receptors, SingleExpSynapse, SpikeTimeParameter, SpikeTimeStimulus, SpikeTimeStimulusParameter, SpikingSynapse, SpikingSynapseParameter, TripodParameter, iSTDPPotential, iSTDPRate, vSTDPParameter,
    PoissonLayer, Stimulus, SpikingSynapse, Population, SNNModel, Poisson, StimulusGroup, LTPParam, STPParam

    export MarkramSTPParameterHet, MarkramSTPParameter

    
    ## Functions from SNNModels
    export asynchronous_state, bin_spiketimes, clear_monitor!, clear_records!, compose, compute_connections, firing_rate, get_time,  load_model, matrix, monitor!, name, place_populations, record, record!, reset_time!, sample_inputs, save_model, set_plasticity!, str_name,     train!, update_spikes!, SNNload, SNNsave, compose, sim!,
    set_plasticity!, change_plasticity!, update_traces!, set_STP!, set_LTP!

    export @update, @update!, @load_units
end
