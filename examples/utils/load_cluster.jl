#!/bin/bash

#=
PROJECTDIR="/pasteur/appa/homes/aquaresi/spiking/network_models" 
OUTDIR=${PROJECTDIR}/logs/out
ERRDIR=${PROJECTDIR}/logs/errors
echo $PROJECTDIR
# =#

#SBATCH --job-name="test_run"            # Job Name
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1                       # 1 CPU allocation per Task
#SBATCH --mem=1GB
#SBATCH --time=1:00:00
#SBATCH --partition=common
#SBATCH -q common
#SBATCH -e ${ERRDIR}/slurm-test_julia_%j.err
#SBATCH -o ${OUTDIR}/slurm-test_julia_%j.out
#SBATCH --qos=fast

#=
srun julia $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
exit
# =#

using Distributed
addprocs(4)
@everywhere using DrWatson
@everywhere using SpikingNeuralNetworks
@everywhere SNN.@load_units;
@everywhere using SNNUtils

# Define the network
@everywhere function create_network()
    network = let
        # Number of neurons in the network
        N = 1000
        # Create dendrites for each neuron
        E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -60mV))
        # Define interneurons 
        I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, El = -50mV))
        # Define synaptic interactions between neurons and interneurons
        E_to_I = SNN.SpikingSynapse(E, I, :ge, p = 0.2, μ = 3.0)
        E_to_E = SNN.SpikingSynapse(E, E, :ge, p = 0.2, μ = 0.5)#, param = SNN.vSTDPParameter())
        I_to_I = SNN.SpikingSynapse(I, I, :gi, p = 0.2, μ = 4.0)
        I_to_E = SNN.SpikingSynapse(
            I,
            E,
            :gi,
            p = 0.2,
            μ = 1,
            param = SNN.iSTDPParameterRate(r = 4Hz),
        )
        norm = SNN.SynapseNormalization(E, [E_to_E], param = SNN.AdditiveNorm(τ = 30ms))

        # Store neurons and synapses into a dictionary
        pop = SNN.@symdict E I
        syn = SNN.@symdict I_to_E E_to_I E_to_E norm I_to_I
        (pop = pop, syn = syn)
    end

    # Create background for the network simulation
    noise = SNN.PoissonStimulus(network.pop[:E], :ge, param = 2.8kHz, neurons = :ALL)
    model = SNN.merge_models(network, noise = noise, silent = true)
    SNN.monitor([model.pop...], [:fire])
    simtime = SNN.Time()
    train!(model = model, duration = 5000ms, time = simtime, dt = 0.125f0)
    path = datadir("example_cluster") |> mkpath
    DrWatson.save(joinpath(path, "network_with_spikes_$(myid()).jld2"), "model", model)
end

for i = 1:nprocs()
    s = @spawnat :any create_network()
end
