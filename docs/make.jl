using Documenter, SpikingNeuralNetworks, .SNNModels, .SNNPlots, .SNNUtils

pages = [
    "Home" => "index.md",
    "Tutorial" => "examples.md",
    "Populations" => "populations.md",
    "Stimuli" => "stimuli.md",
    "Plasticity" => "plasticity.md",
    "API Reference" => "api_reference.md",
    "Models Extension" => "models_ext.md"
]
makedocs(
    sitename = "SpikingNeuralNetworks.jl",
    # sitenameurl = "https://aquaresi.github.io/SpikingNeuralNetworks.jl",

    # strict = [
    #          :doctest,
    #          :linkcheck,
    #          :parse_error,
    #          :example_block,
    #          # Other available options are
    #          # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
    #      ],
         

    modules = [SpikingNeuralNetworks, SNNModels],
     warnonly = [:autodocs_block],
    format = Documenter.HTML(
        # analytics = "UA-90474609-3",
        # assets = ["assets/favicon.ico"],
        # canonical = "https://surrogates.sciml.ai/stable/"
    ),
    pages = pages
    
)
# clean = true,
# assets = ["assets"],
# checkdocs = true,
# doctest = true,
# warnerror = true,
# doctest_setup = """
#     using SpikingNeuralNetworks
#     using SNNPlots
#     SNN.@load_units
# """)
deploydocs(repo = "github.com/JuliaSNN/SpikingNeuralNetworks.jl.git")
