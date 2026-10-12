# Izhikevich, Hodgkin-Huxley, Morris-Lecar

```@meta
CurrentModule = SNNModels
```

Single-compartment models with two exponentially decaying conductances `ge`, `gi` (targets `:ge`,
`:gi`); no `synapse` or `spike` field.

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

Reference: Izhikevich (2003), IEEE Trans. Neural Netw. 14:1569-1572.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IZ(N = 10, param = SNN.IZParameter(a = 0.02, b = 0.2, c = -65, d = 8))  # regular spiking
E.I .= 10
SNN.monitor!(E, [:v, :fire])
SNN.sim!([E]; duration = 200ms, dt = 0.5ms)
```

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

A spike is flagged when ``v`` crosses -20 mV upwards; no reset. Use `dt` of 0.01-0.05 ms.

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

Reference: Hodgkin and Huxley (1952), J. Physiol. 117:500-544; Traub-Miles kinetics as in the
COBAHH benchmark of Brette et al. (2007).

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.HH(N = 5)
E.ge .= 0; E.gi .= 0          # remove the random initial conductances
E.I .= 500pA
SNN.monitor!(E, [:v])
SNN.sim!([E]; duration = 50ms, dt = 0.01ms)
```

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

Reference: Morris and Lecar (1981), Biophys. J. 35:193-213.

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
