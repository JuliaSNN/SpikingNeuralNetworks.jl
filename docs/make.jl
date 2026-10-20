using Documenter, SpikingNeuralNetworks, .SNNModels, .SNNPlots, .SNNUtils

# Every source file of SNNModels, SNNUtils and SNNPlots is rendered by exactly one `@autodocs`
# block (selected with `Pages = [...]`), so that each docstring appears once on the site.
pages = [
    "Home" => "index.md",
    "Getting started" => "getting_started.md",
    "Tutorials" => [
        "Populations" => "tutorials/populations.md",
        "Plasticity" => "tutorials/plasticity.md",
        "Simulations" => "tutorials/simulations.md",
    ],
    "User guide" => [
        "Populations" => "populations.md",
        "Stimuli" => "stimuli.md",
        "Running simulations" => "simulation.md",
        "Recordings" => "recordings.md",
        "Plasticity" => "plasticity.md",
        "Visualization (SNNPlots)" => "visualization.md",
    ],
    "Model catalogue" => [
        "Overview" => "catalogue/index.md",
        "Integrate-and-fire neurons" => "catalogue/neurons_if.md",
        "Izhikevich, Hodgkin-Huxley, Morris-Lecar" => "catalogue/neurons_other.md",
        "Rate models" => "catalogue/rate_models.md",
        "Spike sources" => "catalogue/sources.md",
        "Multicompartment neurons" => "catalogue/multicompartment.md",
        "Synapse and receptor models" => "catalogue/synapses.md",
        "Connections" => "catalogue/connections.md",
        "Plasticity rules" => "catalogue/plasticity_rules.md",
        "Metaplasticity" => "catalogue/metaplasticity.md",
        "Stimuli" => "catalogue/stimuli.md",
        "SNNUtils models" => "catalogue/snnutils_models.md",
        "SNNUtils tools" => "catalogue/snnutils.md",
    ],
    "Performance comparison" => "performance.md",
    "API Reference" => "api_reference.md",
    "Models Extension" => "models_ext.md",
    "Contributing" => "contributing.md",
    "V2 changelog" => "release_notes.md",
]

makedocs(
    sitename = "SpikingNeuralNetworks.jl",
    modules = [SpikingNeuralNetworks, SNNModels, SNNUtils, SNNPlots],
    warnonly = [:autodocs_block],
    format = Documenter.HTML(
        # the API pages are large (all SNNModels docstrings)
        size_threshold = 500 * 2^10,
        size_threshold_warn = 300 * 2^10,
        # the search index covers every docstring of the four packages
        search_size_threshold_warn = 1000 * 2^10,
    ),
    pages = pages,
)

deploydocs(repo = "github.com/JuliaSNN/SpikingNeuralNetworks.jl.git")
