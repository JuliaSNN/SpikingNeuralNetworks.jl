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
- With `c` chunks on `k` threads the parallel stages take `ceil(c / k) / c` of the serial time, so the best
  speed-up is `c / ceil(c / k)`. 8 chunks on 4 threads gives 4, on 6 threads still 4; 5 chunks on 4 threads gives
  2.5; 6 chunks give 3. Choose `neurons_per_thread` so that the number of chunks is a multiple of the thread count
  (4 threads, N = 6000: 1536 gives 4 chunks, 768 gives 8) or let the tuner pick it.
- Large synapse counts are also limited by memory bandwidth: the normalisation of 25 M synapses at 4 threads moves about
  30 GB/s, close to the 50 GB/s that 12 threads of this workstation reach [Likely].
- Weights of event-driven `vSTDP` synapses are materialised serially before each normalisation step; a normalisation
  of such synapses gains little from threads.
- Threaded runs use up to 8 cores for as long as they run, even when the chunks are small, because the
  workers spin. Give them dedicated cores. A worker that the operating system takes away stalls the
  whole step: on a shared machine this showed as occasional runs 3 to 7 times slower. Pin the process
  (`taskset`, one core per physical core) or request whole cores from the scheduler.
- `Threads.@spawn` user tasks that each step their own model work and give the same results as serial
  runs; Polyester hands a free worker to each and falls back to the calling thread when none is free.

## Theoretical speed-up

Amdahl bound for the part of the step that is parallel, from the share of the serial step measured
with a sampling profile on one thread (`validation/threading/out/profile_split_*.txt`, the first four
rows of each table: balanced E/I network of `IF` neurons, `sim!` or `train!` with event-driven
`STDP` on E to E; `AdEx`: share of the population stages in an N = 10 000 network). With parallel fraction `f` and `k` threads the bound is
`1 / ((1 - f) + f / k)`; `k` stops counting at 8 because a population has at most 8 chunks. The bound ignores the
fork and join (0.2 to 0.7 us per fork) and the cache traffic between cores, so a measured speed-up is lower.

**Populations** (neuron update and synaptic-variable update; `f` = share of the serial step)

| Network | f | 2 threads | 4 threads | 8 threads | Measured, 4 / 8 threads |
|---|---|---|---|---|---|
| `IF`, N = 10 000 | 0.49 | 1.32 | 1.58 | 1.75 | 1.03 / 1.01 [from report] |
| `IF`, N = 2000 | 0.43 | 1.27 | 1.48 | 1.60 | 0.4 to 1.08 (below one chunk of 512: serial) [from report] |
| `IF` + STDP, N = 10 000 | 0.36 | 1.22 | 1.37 | 1.46 | not measured |
| `AdEx`, N = 10 000 | 0.78 | 1.64 | 2.41 | 3.15 | 1.35 / 1.47 [from report] |

**Synapses** (`forward!`, delay arrival and event-driven plasticity, partitioned by postsynaptic range)

| Network | f | 2 threads | 4 threads | 8 threads | Measured |
|---|---|---|---|---|---|
| `forward!` only, `IF` N = 10 000 | 0.35 | 1.21 | 1.35 | 1.43 | not measured alone |
| `forward!` only, `IF` N = 2000 | 0.20 | 1.11 | 1.18 | 1.21 | not measured alone |
| `forward!` + STDP, N = 10 000 | 0.48 | 1.31 | 1.56 | 1.72 | not measured alone |
| `forward!` + STDP, N = 2000 | 0.30 | 1.17 | 1.29 | 1.35 | not measured alone |
| `SynapseNormalization`, 4.6 M / 25 M synapses | 0.8 to 0.9 [Likely] | - | 1.9 / 2.2 | 2.2 (6 threads) | 1.9 / 2.2 at 4 threads [measured, `bench/normalisation_bench.jl`] |

Populations and synapses together (`IF`, N = 10 000): `f` = 0.83 gives 1.71 / 2.66 / 3.69 at 2 / 4 / 8 threads;
at N = 2000 `f` = 0.63 gives 1.46 / 1.90 / 2.23. The ceiling with unlimited threads is 6.0 and 2.7.
Reading: networks of simple neurons are bounded by the serial remainder (stimuli, spike lists, clock,
recording) and by the cost of one fork per stage; heavy neurons and large synapse counts have
the largest bound and are the ones that gain. The normalisation pass is limited by memory bandwidth before it
reaches its bound.

## Threads or processes

A *thread* shares the memory of one Julia process: with `julia -t 8` one simulation uses eight cores, each
working on a chunk of the same populations. A *process* is a separate Julia instance with its own copy of
the model: eight processes run eight different simulations at once and do not communicate.

This matters for Optuna. Every trial is an independent simulation, so the natural parallelism of a search
is one single-thread process per trial (Optuna workers, SLURM array jobs): no fork cost, no spinning workers
and no serial remainder, so the throughput grows with the number of processes while the working sets fit in the
cache. Large working sets share the memory bandwidth: 12 independent processes reach 11.9 times the
throughput of one process on small models and 4.6 times on large ones
[measured, `cpu_decision/out/summary_throughput.csv`]. Threads make one trial finish sooner, not more trials
finish per hour; they pay when one simulation is large enough to pass the tables above (heavy neurons, tens of
thousands of neurons or millions of synapses) and its run time is the bottleneck.

A job that mixes both (`-t 4` per trial, several trials at once) needs `trials x threads` cores in the cluster
request; otherwise workers compete for cores and, because they spin, the runs become slower than serial.

## Threading and results

Results are bit-identical for every chunk size and every thread count; the test suite compares threaded
and serial runs bit for bit, including `train!` with plasticity and normalisation. A saved model does not
store the setting: it belongs to the run, not to the model.

## Reference

The docstrings of `neurons_per_thread`, `foreach_chunk`, `set_optimal_neurons_per_thread!` and the other
threading functions are in [Running simulations](simulation.md#API).
