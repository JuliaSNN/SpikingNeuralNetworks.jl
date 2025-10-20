
The library also allows to define an arbitrary complex noise function with the `CurrentVariableParameter` type. In this case we must define a function, in this case `sinusoidal_current`, which is called runtime to determine the input current to each neuron in the population, the function must accept three arguments: 
1. a dictionary with the `variables::Dict`;
2. the time of the model `t::Float32`;
3. the index of the neuron `i::Int32`.

We thus define the set of variables that the function uses to determine the current and pass them along the function to `CurrentVariableParameter`. 

In the following example we used a plain sinusoidal current that stimulate the two neurons in the population with a phase a frequency of 1Hz and a phase shift of `3/4 π`

```julia 


# Create a populations with 2 IF neurons
E = SNN.IF(; N = 2, 
    param=if_parameter,
    )
SNN.monitor!(E, [:v, :fire, :w, :I], sr = 2kHz)

# Create a withe noise input current 

function sinusoidal_current(variables::Dict, t::Float32, i::Int)
    # Extract the parameters from the variables dictionary
    amplitude = variables[:amplitude]
    frequency = variables[:frequency]
    phase = variables[:shift_phase]
    
    # Calculate the current value at time t for neuron i
    return amplitude * sin(2 * π * frequency * t + i*phase)
end

variables = Dict(
    :amplitude => [50pA],
    :frequency => [1Hz],
    :shift_phase => [π*3/4], # Phase shift for each neuron
)

current_param = SNN.CurrentVariableParameter(variables, sinusoidal_current)
current = CurrentStimulus(E, :I, param=current_param)
model = compose(; E = E, I=current)
SNN.sim!(; model, duration = 2000ms)

p = plot(
    vecplot(E, :v, add_spikes=true, ylabel="Membrane potential (mV)", ylims=(-80, 10)),
    vecplot(E, :I, ylabel="External current (pA)", c=:gray, lw=0.4, alpha=0.4),
    layout=(
        2, 1
    ),
    size=(600, 500),
    xlabel= "Time (s)",
    leftmargin=10Plots.mm,
)

```

![Variable input current](assets/examples/variable_current.png)
