using DrWatson 
quickactivate("/home/user/mnt/zeus/User_folders/aquaresi/network_models")
using SpikingNeuralNetworks, Documenter

makedocs(sitename = "Julia SpikingNeuralNetworks",
        # # sitenameurl = "https://aquaresi.github.io/SpikingNeuralNetworks.jl",
        modules = [SpikingNeuralNetworks])
        # pages = [
        #     "Home" => "index.md",
        #     "Installation" => "installation.md",
        #     "Getting Started" => "getting_started.md",
        #     "Examples" => "examples.md",
        #     "API Reference" => "api_reference.md",
        #     "Contributing" => "contributing.md"
        # ],
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
