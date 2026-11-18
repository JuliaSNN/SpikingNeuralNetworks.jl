# Izhikevich, Hodgkin-Huxley, Morris-Lecar

```@meta
CurrentModule = SNNModels
```

Single-compartment models with two exponentially decaying conductances `ge`, `gi` (targets `:ge`,
`:gi`); no `synapse` or `spike` field.

| Model | Reference | Integration |
|:------|:----------|:------------|
| `IZ` | Izhikevich (2003), IEEE Trans. Neural Netw. 14:1569-1572, [10.1109/TNN.2003.820440](https://doi.org/10.1109/TNN.2003.820440) | forward Euler, two half steps of `dt/2` for the quadratic part of `v` |
| `HH` | Hodgkin and Huxley (1952), J. Physiol. 117:500-544, [10.1113/jphysiol.1952.sp004764](https://doi.org/10.1113/jphysiol.1952.sp004764); Traub-Miles kinetics as in Brette et al. (2007), J. Comput. Neurosci. 23:349-398, [10.1007/s10827-007-0038-6](https://doi.org/10.1007/s10827-007-0038-6) | forward Euler, gating variables first, then `v` |
| `MorrisLecar` | Morris and Lecar (1981), Biophys. J. 35:193-213, [10.1016/S0006-3495(81)84782-0](https://doi.org/10.1016/S0006-3495(81)84782-0) | forward Euler, `v` then `w` |

The quadratic term of `IZ` gives the model its two-variable form: ``v`` is the membrane potential, ``u`` a recovery variable
that sets the firing pattern through ``a, b, c, d``.

## Izhikevich model: `IZ`

```math
\begin{aligned}
\frac{dv}{dt} &= 0.04 v^2 + 5 v + 140 - u + I + g_e (E_e - v) + g_i (E_i - v) \\
\frac{du}{dt} &= a\,(b v - u) \\
\frac{dg_e}{dt} &= -\frac{g_e}{\tau_e}, \qquad \frac{dg_i}{dt} = -\frac{g_i}{\tau_i}
\end{aligned}
```

When ``v > 30`` mV: ``v \leftarrow c``, ``u \leftarrow u + d``. ``I`` is in mV/ms.

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `a` | `0.01` | 1/ms | recovery time scale |
| `b` | `0.2` | - | sensitivity of ``u`` to ``v`` |
| `c` | `-65` | mV | reset value of ``v`` |
| `d` | `2` | - | reset increment of ``u`` |
| `τe` | `5ms` | ms | decay of `ge` |
| `τi` | `10ms` | ms | decay of `gi` |
| `Ee` | `0mV` | mV | excitatory reversal potential |
| `Ei` | `-80mV` | mV | inhibitory reversal potential |

**Integration:** conductances decay by forward Euler; `v` takes two forward-Euler half steps of `dt/2`; `u` one Euler step with the new `v`;
then the synaptic term `dt (ge (Ee - v) + gi (Ei - v))` is added; spikes are detected at `v > 30` and reset. No refractory period.

**Reference:** Izhikevich (2003), IEEE Trans. Neural Netw. 14:1569-1572, [10.1109/TNN.2003.820440](https://doi.org/10.1109/TNN.2003.820440).

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IZ(N = 10, param = SNN.IZParameter(a = 0.02, b = 0.2, c = -65, d = 8))  # regular spiking
E.I .= 10
SNN.monitor!(E, [:v, :fire])
SNN.sim!([E]; duration = 200ms, dt = 0.5ms)
```

!!! note "Implementation note"
    `ge` and `gi` are in 1/ms (``\tau_m = 1/(g_e + g_i)``) and enter an explicit step: the step is stable only for `dt (ge + gi) < 2`,
    and a negative conductance is unstable at any `dt`. They start at zero.

```@autodocs
Modules = [SNNModels]
Pages   = ["populations/iz.jl"]
```

## Hodgkin-Huxley model: `HH`

```math
\begin{aligned}
C_m \frac{dv}{dt} &= I + g_l (E_l - v) + g_e (E_e - v) + g_i (E_i - v)
                    + g_n m^3 h (E_n - v) + g_k n^4 (E_k - v) \\
\frac{dx}{dt} &= \alpha_x(v)\,(1 - x) - \beta_x(v)\, x, \qquad x \in \{m, n, h\}
\end{aligned}
```

with ``u = v - V_t`` (mV) and rates in 1/ms:

```math
\begin{aligned}
\alpha_m &= \frac{0.32\,(13 - u)}{e^{(13 - u)/4} - 1}, &
\beta_m &= \frac{0.28\,(u - 40)}{e^{(u - 40)/5} - 1}, \\
\alpha_n &= \frac{0.032\,(15 - u)}{e^{(15 - u)/5} - 1}, &
\beta_n &= 0.5\, e^{(10 - u)/40}, \\
\alpha_h &= 0.128\, e^{(17 - u)/18}, &
\beta_h &= \frac{4}{1 + e^{(40 - u)/5}}.
\end{aligned}
```

A spike is flagged when ``v`` crosses -20 mV upwards (one flag per action potential); no reset.

| Field | Default | Value | Meaning |
|:------|:--------|:------|:--------|
| `Cm` | `1uF * cm^(-2) * 20000um^2` | 200 pF | membrane capacitance |
| `gl` | `5e-5siemens * cm^(-2) * 20000um^2` | 10 nS | leak conductance |
| `El` | `-65mV` | | leak reversal potential |
| `Ek` | `-90mV` | | potassium reversal potential |
| `En` | `50mV` | | sodium reversal potential |
| `gn` | `100msiemens * cm^(-2) * 20000um^2` | 20000 nS | maximal sodium conductance |
| `gk` | `30msiemens * cm^(-2) * 20000um^2` | 6000 nS | maximal potassium conductance |
| `Vt` | `-63mV` | | offset of the gating kinetics (not a threshold) |
| `τe`, `τi` | `5ms`, `10ms` | | decay of `ge`, `gi` |
| `Ee`, `Ei` | `0mV`, `-80mV` | | synaptic reversal potentials |

**Integration:** forward Euler, sequential: `m`, `n`, `h`, then `v` with the new gating variables, then `ge`, `gi` decay.

**Reference:** Hodgkin and Huxley (1952), J. Physiol. 117:500-544, [10.1113/jphysiol.1952.sp004764](https://doi.org/10.1113/jphysiol.1952.sp004764); Traub-Miles kinetics as in the
COBAHH benchmark of Brette et al. (2007), J. Comput. Neurosci. 23:349-398, [10.1007/s10827-007-0038-6](https://doi.org/10.1007/s10827-007-0038-6).

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.HH(N = 5)
E.ge .= 0; E.gi .= 0          # remove the random initial conductances
E.I .= 500pA
SNN.monitor!(E, [:v])
SNN.sim!([E]; duration = 50ms, dt = 0.01ms)
```

!!! warning "Implementation note"
    The forward-Euler step is stiff (``g_n/C_m = 100`` 1/ms): it needs `dt <= 0.02 ms`. The default `dt = 0.125 ms` of `sim!` gives `NaN`; pass
    `dt = 0.01ms`. The `ge`, `gi` start random, set them to zero as in the example. The rate functions use their limit at the 0/0 points.

```@autodocs
Modules = [SNNModels]
Pages   = ["populations/hh.jl"]
```

## Morris-Lecar model: `MorrisLecar`

```math
\begin{aligned}
C_m \frac{dv}{dt} &= I + g_l (E_l - v) + g_{Ca}\, m_\infty(v) (E_{Ca} - v) + g_K\, w\, (E_K - v)
                    + g_e (E_e - v) + g_i (E_i - v) \\
\frac{dw}{dt} &= \frac{w_\infty(v) - w}{\tau_w(v)}
\end{aligned}
```

```math
m_\infty(v) = \tfrac12 \left(1 + \tanh\frac{v - V_1}{V_2}\right), \quad
w_\infty(v) = \tfrac12 \left(1 + \tanh\frac{v - V_3}{V_4}\right), \quad
\tau_w(v) = \frac{1}{\phi \cosh\left(\frac{v - V_3}{2 V_4}\right)}
```

A spike is flagged when ``v`` crosses 20 mV upwards; no reset.

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `Cm` | `6.69pF` | pF | membrane capacitance |
| `El`, `EK`, `ECa` | `-50mV`, `-70mV`, `100mV` | mV | reversal potentials |
| `gl`, `gK`, `gCa` | `0.5nS`, `2nS`, `1.1nS` | nS | leak, potassium, calcium conductances |
| `τe`, `τi` | `5ms`, `10ms` | ms | decay of `ge`, `gi` |
| `V1`, `V2` | `30mV`, `15mV` | mV | calcium activation midpoint and slope |
| `V3`, `V4` | `0mV`, `30mV` | mV | potassium activation midpoint and slope |
| `ϕ` | `25Hz` | 1/ms | potassium rate scale (0.025 per ms) |
| `Ee`, `Ei` | `0mV`, `-75mV` | mV | synaptic reversal potentials |

**Integration:** forward Euler, sequential: `v` with the old `w`, then `w` with the new `v`, then the synaptic term
`dt (ge (Ee - v) + gi (Ei - v)) / Cm`, then `ge`, `gi` decay.

**Reference:** Morris and Lecar (1981), Biophys. J. 35:193-213, [10.1016/S0006-3495(81)84782-0](https://doi.org/10.1016/S0006-3495(81)84782-0).

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.MorrisLecar(N = 5)
E.I .= 100pA
SNN.monitor!(E, [:v, :w])
SNN.sim!([E]; duration = 200ms, dt = 0.05ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["populations/morrislecar.jl"]
```
