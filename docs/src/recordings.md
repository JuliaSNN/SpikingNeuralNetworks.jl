# Recordings

One of the strengths of the **SpikingNeuralNetworks.jl** library is its easy access to all network variables. You can record any dynamic variable used at runtime, and to optimize memory usage, recordings can be subsampled.

To demonstrate how recordings work, let’s instantiate a network model with excitatory and inhibitory recurrent connections. Excitatory connections follow **short-term plasticity (STP)**, while inhibitory connections use **long-term plasticity (LTP)**. We will show how to record different types of variables simulated in the network.

```julia
using SpikingNeuralNetworks
using Statistics
SNN.@load_units

# AdEx neuron with fixed external current connections with multiple receptors
E = SNN.AdEx(; N = 800, param = SNN.AdExParameter(; El = -50mV))
I = SNN.IF(; N = 200, param = SNN.IFParameter())
EE = SNN.SpikingSynapse(E, E, :he; μ = 2, p = 0.02, STPParam = SNN.MarkramSTPParameter())
EI = SNN.SpikingSynapse(E, I, :ge; μ = 30, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :hi; μ = 50, p = 0.02, LTPParam = SNN.iSTDPRate(r=5Hz))
II = SNN.SpikingSynapse(I, I, :gi; μ = 10, p = 0.02)
model = SNN.compose(; E, I, EE, EI, IE, II)
```

To monitor any model variable, use the [`monitor!`](@ref SNN.SNNModels.monitor!) function. This function takes the component instance (e.g., `E`) and the symbol (or list of symbols) you want to record. Optionally, you can specify the sampling rate (`sr`, default: 1kHz) for the recording.

---

## Population Variables

First, let’s record variables associated with populations. We will record the excitatory and inhibitory conductances (`:ge`, `:gi`), firing rate (`:fire`), and membrane potential (`:v`) for all populations in the network model.

```julia
SNN.monitor!(E, [:ge, :gi], sr=200Hz)
SNN.monitor!(model.pop, :v, sr=200Hz)
SNN.monitor!(model.pop, :fire)
SNN.sim!(model = model; duration = 5second)
```

To access recorded variables, use the [`record`](@ref SNN.SNNModels.record) function. This function takes the network component and the variable of interest as arguments. It returns an array (neurons × time), interpolated over the interval defined by `:start_time` and `:end_time` (the model’s time when `monitor!` was called and the last time point of the simulation). The resolution of the recording is determined by the sampling rate. Thanks to interpolation, you can access the variable at any continuous time point.

```julia
v = SNN.record(model.pop.E, :v)
@info "V is: type $(nameof(typeof(v))), size $(size(v))"
v[1, 3.14s]
v[1:10, 2.4s:15ms:3.1s]
v, r = SNN.record(model.pop.E, :v, range=true)
@info "V is: type $(nameof(typeof(v))), size $(size(v)), r size: $(size(r))"
v = SNN.record(model.pop.E, :v, interpolate=false)
@info "V is: type $(nameof(typeof(v))), size $(size(v))"
```

!!! note
    Currently, it is not possible to deactivate recordings while keeping the variable in the monitored pool. This behavior may change in future updates.

---

### Spiketimes and Firing Rates

Spiketimes are stored as `SNN.Spiketimes`, a `Vector` of `Vector`. The first vector contains the spiketimes of each neuron in milliseconds (neurons × times).

```julia
# Spiketimes
spiketimes = SNN.spiketimes(model.pop.E) # All spiketimes
@info "Spiketimes is: type $(nameof(typeof(spiketimes))), size $(size(spiketimes)), neuron 1 has $(length(spiketimes[1])) spikes"

spiketimes = SNN.spiketimes(model.pop.E; interval=0:1ms:5second) # Spiketimes in the specified interval
@info "Spiketimes is: type $(nameof(typeof(spiketimes))), size $(size(spiketimes)), neuron 1 has $(length(spiketimes[1])) spikes"
```

For convenience, you can also access binned spikes using `bin_spiketimes(comp<:AbstractPopulation; interval::AbstractRange)`. This function returns a tuple: a matrix (neurons × bins) where each entry represents the number of spikes in that bin, and the `interval` range. The spiketimes are binned within the extremes of `interval`, with the bin width defined by the `interval` step.

```julia
# Binned spikes
interval = 0:10ms:5s # 
bins, r = SNN.bin_spiketimes(model.pop.E; interval)
@info "Bins is: type $(nameof(typeof(bins))), size $(size(bins)), r size: $(size(r))"
```

To directly access the firing rate, use `fr, r = SNN.firing_rate(model.pop.E; interval::AbstractRange)`. The firing rate is an interpolated array that samples a continuous firing rate signal at the time points defined by `interval` (a mandatory keyword argument). The continuous signal is obtained by convolving the binned spike train with an alpha-function kernel (time constant τ, default: 10ms). The firing rate is returned as a matrix (neurons × time points), where each entry represents the firing rate in Hz at that time point.

```julia
# Firing rate
fr, r = SNN.firing_rate(model.pop.E; interval) # Interpolated firing rate
@info "Fr is: type $(nameof(typeof(fr))), size $(size(fr)), r size: $(size(r))"
fr, r = SNN.firing_rate(model.pop.E; interval, interpolate=false) # Non-interpolated firing rate
@info "Fr is: type $(nameof(typeof(fr))), size $(size(fr)), r size: $(size(r))"
```

You can also access the firing rate for the entire population:

```julia
fr, r, pop_names = SNN.firing_rate(model.pop; interval)
```

For simplicity, you can also access firing rates and spike times via the `record` function:

```julia
fr = SNN.record(model.pop.E, :fire; interval)
@info "Fr is: type $(nameof(typeof(fr))), size $(size(fr))"
fr, r = SNN.record(model.pop.E, :fire; interval, range=true)
@info "Fr is: type $(nameof(typeof(fr))), size $(size(fr)), r size: $(size(r))"
SNN.record(model.pop.E, :spikes)
```

!!! note
    The model instance declared in the `Main` scope (`E`) and the instance in the network model (`model.pop.E`) point to the same object in memory. Operating on either is equivalent.

!!! note
    Recorded variables are stored in the component’s `records` field. The storage method is non-trivial and subject to future changes, so we avoid detailing it here.

---

## Synaptic Variables

We now add to the recordings the synaptic strength (`:W`) and efficacy (`:ρ`) for the inhibitory and excitatory connections. We also record the variables (`:x` and `:u`) for the STP in the excitatory connections and the filtered post-synaptic trace of the inhibitory STDP (`:tpost`). When recording plasticity variables, you must specify which set of variables you are referring to. This can be done using the keyword argument `variables` or implicitly by adding a third positional argument to the `monitor!` function.

```julia
SNN.monitor!(EE, [:ρ], sr=10Hz)
SNN.monitor!(EI, [:W], sr=10Hz)
SNN.monitor!(IE, [:tpost]; sr=10Hz, variables=:LTPVars)
SNN.monitor!(EE, [:x, :u], :STPVars; sr=10Hz)
SNN.train!(model = model; duration = 5second)
```

!!! warning
    Recording synaptic strength or efficacy can be memory-intensive in large networks. We recommend using a low sampling rate.

!!! note
    `STPVars` and `LTPVars` are special keywords representing sets of short-term and long-term plasticity-related variables, respectively.

---

### Synaptic Connectivity

Synaptic connectivity is stored in a sparse format as a matrix with dimensions `(N_post, N_pre)`. You can always access the synaptic weights of the connections directly:

```julia
W = SNN.matrix(EE)  # Default: returns the synaptic strength matrix at the last time point
W = SNN.matrix(EE, :W)
ρ = SNN.record(EE, :ρ)
```

#### Accessing Pre- and Post-Synaptic Neurons

You can access the pre- and post-synaptic neurons for a single neuron or a set of neurons:

**Single Neuron**
```julia
neuron = 1
Is = SNN.postsynaptic(EE, neuron)  # Post-synaptic neurons
mean(W[Is, neuron])  # Mean synaptic weight of post-synaptic connections
Js = SNN.presynaptic(EE, neuron)  # Pre-synaptic neurons
mean(W[neuron, Js])  # Mean synaptic weight of pre-synaptic connections
```

**Multiple Neurons**
```julia
neurons = 1:10
W = SNN.matrix(EE)
Is = SNN.presynaptic(EE, neurons)  # Pre-synaptic neurons for multiple neurons
Js = SNN.postsynaptic(EE, neurons)  # Post-synaptic neurons for multiple neurons
```

#### Synaptic Weight Matrices

When recorded, the matrix of synaptic weights or synaptic efficacy can be obtained using the `record` function. The returned value is a sparse matrix in a vector format, where only the non-zero values are maintained.

**Get the sparse vector `ρ` at time point `t`:**

This returns only the non-zero elements of the matrix.
```julia
ρ, r = SNN.record(EE, :ρ, range=true)
histogram(ρ[:, 6.5s])
```

**Reconstruct the full matrix from the sparse vector `ρ` at time point `t`:**

This operation reverses the sparse representation and returns the full matrix. You can pass either the vector obtained from `SNN.record` or the synapse object and the symbol of the variable.
```julia
ρ_mat1 = SNN.matrix(EE, ρ, 6.5s)
ρ_mat2 = SNN.matrix(EE, :ρ, 6.5s)
all(ρ_mat1 .== ρ_mat2)  # true
```

**Get the matrix at multiple time points:**
This returns a 3D array of size `(N_E, N_E, T)`, where `T` is the number of time points in the specified range.
```julia
ρ_T1 = SNN.matrix(EE, :ρ, 6.5s:10ms:7s)
ρ_T2 = SNN.matrix(EE, ρ, 6.5s:10ms:7s)
```

!!! tip
    For visualization, you can use the functions defined in [Plots](@ref) or use packages like `Plots.jl` to plot recorded variables or  
    ```julia
    using Plots
    plot(r, v[1,:], label="Membrane potential of neuron 1")
    ```

---

### Plasticity Variables

Plasticity-related variables, such as STP (`:x`, `:u`) or LTP (`:tpost`), can also be accessed using the `record` function by adding the name of the set of variables of interest (`STPVars` or `LTPVars`) as a prefix. For example, to retrieve the STP variables for the synapse `EE`:

```julia
x = SNN.record(EE, :STPVars_x)
@info "x is: type $(nameof(typeof(x))), size $(size(x))"
x[1, 3.14s]
x[1:10, 2.4s:15ms:3.1s]
x, r = SNN.record(EE, :STPVars_x, range=true)
@info "x is: type $(nameof(typeof(x))), size $(size(x)), r size: $(size(r))"
x = SNN.record(EE, :STPVars_x, interpolate=false)
@info "x is: type $(nameof(typeof(x))), size $(size(x))"
```

Similarly, for LTP variables in the synapse `IE`:

```julia
tpost = SNN.record(IE, :LTPVars_tpost)
@info "tpost is: type $(nameof(typeof(tpost))), size $(size(tpost))"
tpost[1, 3.14s]
tpost[1:10, 2.4s:15ms:3.1s]
tpost, r = SNN.record(IE, :LTPVars_tpost, range=true)
@info "tpost is: type $(nameof(typeof(tpost))), size $(size(tpost)), r size: $(size(r))"
tpost = SNN.record(IE, :LTPVars_tpost, interpolate=false)
@info "tpost is: type $(nameof(typeof(tpost))), size $(size(tpost))"
```

!!! note
    High sampling rates or recording many variables simultaneously can impact performance. Use subsampling (`sr` keyword) to balance memory usage and resolution.
