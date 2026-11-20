# Threads and multicore

```@meta
CurrentModule = SNNModels
```

After this page you can switch threading on for a simulation, know which parts of a step run in
parallel, choose the chunk size, and decide when to use threads and when to use several processes.

Threading is **off by default**. With one Julia thread, or with `neurons_per_thread = 0`, the serial
loops run and no worker thread is touched, so the cost of the feature is zero.

## Switch it on

Two settings are needed: Julia must have several threads, and the model needs a chunk size.

```bash
julia -t 8                 # or JULIA_NUM_THREADS=8, or -t auto
```

<!-- norun -->
```julia
using SpikingNeuralNetworks
SNN.@load_units

SNN.sim!(model, 1s; neurons_per_thread = 512)      # this run only (also train!)
SNN.set_neurons_per_thread!(512)                   # every following run
SNN.neurons_per_thread()                           # 512
SNN.set_neurons_per_thread!(0)                     # off
SNN.with_neurons_per_thread(512) do                # temporary setting
    SNN.sim!(model, 1s)
end
SNN.set_optimal_neurons_per_thread!(model)         # time a copy of the model, keep the fastest setting
```

`neurons_per_thread = 0` or `Inf` is off. The tuner times a copy of the model for 128, 256, 512,
1024, 2048 neurons per chunk and off, prints the table, and keeps the fastest. It leaves `model`
untouched and does nothing with one thread. Run it on the machine and with the thread count of the
production run: the optimum depends on both. A tie with "off" means threading does not help this model.

## How the work is split

- A population of `N` neurons is cut into batches of 128 neurons. A chunk is a whole number of
  batches, so chunk boundaries are multiples of 128 (no two chunks write to the same cache line of a
  `Float32` array).
- The chunk size is `neurons_per_thread` rounded to whole batches (500 gives 512). There are at
  most 8 chunks per population: when the setting would give more, the chunks are made larger.
  A population smaller than one chunk runs serially.
- The chunks depend on `N` and the setting only, never on the number of threads. Two threads or
  eight give the same chunks, the same arithmetic order and bit-identical results. Changing the
  setting does not change the results either.
- Chunks run on persistent Polyester workers (`@batch`): one fork and join costs 0.2 to 0.7 us.
  The workers spin while a threaded run is going and sleep when it ends.
- `foreach_chunk(f, N)` (macro form `@chunks N rng body`) gives the same chunking to your own loops:
  `f(range)` is called once per chunk, `f` must write only the entries of its range and must not allocate.

## What runs in parallel

| Part of the step | Threaded | Notes |
|---|---|---|
| Neuron update and synaptic-variable update of `IF`, `AdEx`, `Identity`, `BallAndStick`, `Tripod`, `Multipod` (and the `PaperScheme` variants) | yes | by chunk of neurons |
| Other populations (`IZ`, `HH`, `MorrisLecar`, rate models) | no | serial |
| `forward!` of `SpikingSynapse` | yes | by postsynaptic range, each target receives its additions in serial order |
| Event-driven `STDP`, `iSTDP` (rate) and short-term plasticity of `SpikingSynapse` | yes | same partition |
| `SynapseNormalization` (sums and update) | yes | sums by postsynaptic range, update by blocks of synapses |
| `vSTDP`, `iSTDPPotential`, `AggregateScaling` | no | serial |
| Stimuli (Poisson sources, noise), recording, clock | no | serial |
| Spike lists | no | serial, built before the parallel stages |

A synapse with fewer than `SNNModels.SYN_MIN_OPS[]` (default 8000) synaptic operations in a step
(spikes times fan-out) runs serially: the fork costs more than it saves. `SynapseNormalization` uses
the same threshold on its number of synapses.

## Limits

**It does not always pay.** The serial parts bound the gain (Amdahl): at the sizes below the neuron
update is 20 to 30 % of a step of a network of simple neurons, and the stages that stay serial
(stimuli, recording, `vSTDP`) are the rest. Every fork and join costs time and moves cache lines
between cores.

| Case (8 threads unless noted) | Speed-up | Source |
|---|---|---|
| `IF` balanced E/I network, N up to 10 000 | at most 1.03 (no gain) | [from report] `validation/threading/README.md` |
| `AdEx` network, N = 20 000 | 2.2 | [measured, `final_v2/out/t12.csv`] |
| `Tripod` network, NE = 4000 | 1.4 | [measured, `final_v2/out/t12.csv`] |
| `AdEx` + event-driven vSTDP, N = 10 000 / 50 000 (12 threads) | 1.3 / 2.0 | [measured, `cpu_decision/out/summary_points.csv`] |
| `Tripod` network, NE = 2000 / 10 000 (12 threads) | 1.1 / 2.2 | [measured, `cpu_decision/out/summary_points.csv`] |
| `SynapseNormalization`, 4.6 M / 25 M synapses, 4 threads, 512 neurons per chunk, against the serial pass | 1.9 / 2.2 | [measured, `SNNModels/bench/normalisation_bench.jl`] |

Rules read from these numbers:

- Heavy neurons (`AdEx`, `Tripod`, `Multipod`) and large networks gain; plain `IF`
  networks and networks below a few thousand neurons do not. Test with the tuner before a production run.
- The gain is flat beyond 8 threads, because a population has at most 8 chunks. With 2 threads the
  threaded normalisation of 4.6 M synapses is 8 % slower than serial; from 4 threads it gains.
- The large cases stop improving around 4 to 8 threads. The pass is limited by memory bandwidth
  (12 threads of this workstation reach 50 GB/s) [Likely].
- Threaded runs use up to 8 cores for as long as they run, even when the chunks are small, because the
  workers spin. Give them dedicated cores. A worker that the operating system takes away stalls the
  whole step: on a shared machine this showed as occasional runs 3 to 7 times slower. Pin the process
  (`taskset`, one core per physical core) or request whole cores from the scheduler.
- `Threads.@spawn` user tasks that each step their own model work and give the same results as serial
  runs; Polyester hands a free worker to each and falls back to the calling thread when none is free.

## Threads or processes

For parameter searches (Optuna trials, seeds, sweeps) run one single-thread process per trial: no
thread overhead and no spinning workers, and the scaling is close to linear while the working sets fit
in the cache. Large working sets share the memory bandwidth: 12 independent processes reach 11.9
times the throughput of one on small models and 4.6 times on large ones
[measured, `cpu_decision/out/summary_throughput.csv`]. Use threads when one simulation is too slow
and the model is large enough to pass the table above, or when a single long run must finish sooner.

A job that mixes both (`-t 4` threads per trial, several trials at once) needs the core count of
the cluster job to match `trials x threads`; otherwise workers compete and the spinning makes it slower than
serial.

## Threading and results

Results are bit-identical for every chunk size and every thread count; the test suite compares threaded
and serial runs bit for bit, including `train!` with plasticity and normalisation. A saved model does not
store the setting: it belongs to the run, not to the model.

## Reference

The docstrings of `neurons_per_thread`, `foreach_chunk`, `set_optimal_neurons_per_thread!` and the other
threading functions are in [Running simulations](simulation.md#API).
