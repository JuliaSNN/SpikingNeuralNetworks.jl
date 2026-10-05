# SNNUtils: protocols and analysis

```@meta
CurrentModule = SNNUtils
```

[SNNUtils.jl](https://github.com/JuliaSNN/SNNUtils.jl) provides stimulation protocols and analysis
tools on top of SNNModels. It is a dependency of `SpikingNeuralNetworks`, which re-exports only
`SVCtrain`; load it explicitly with `using SNNUtils` to access the functions below. The
parameter collections of `src/models/` are described in [SNNUtils models](snnutils_models.md).

## Word and phoneme sequences

A *lexicon* is a `NamedTuple` `(dict, symbols, ph_duration, silence)` built from a dictionary
word => phonemes; a *sequence* adds a 6-row matrix with, for each element, its word, phoneme,
duration (ms), type (`:onset`, `:offset`, `:offset_j`, `:silence`), onset and offset time (ms), and
the row index `line_id`.

- [`getdictionary`](@ref), [`getphonemes`](@ref), [`getduration`](@ref), [`generate_lexicon`](@ref)
  and the shortcut [`get_lexicon`](@ref) build lexicons.
- [`generate_sequence`](@ref) builds the timed sequence from a generator function;
  [`word_phonemes_sequence`](@ref) is the generator provided (modes `:fixed`, `:random`,
  `:balanced`). [`generate_balanced_sequence`](@ref) shuffles a balanced list of sounds.
- [`sign_intervals`](@ref), [`all_intervals`](@ref), [`merge_intervals`](@ref),
  [`time_in_interval`](@ref), [`start_interval`](@ref), [`sequence_end`](@ref) query the timing.
- [`stimuli_names`](@ref) (`w_`/`p_` prefixed names) and [`symbol_names`](@ref) return the
  symbols; [`getstimsym`](@ref), [`getstim`](@ref), [`getneurons`](@ref) look up stimuli.

```julia
using SpikingNeuralNetworks, SNNUtils
SNN.@load_units
lexicon = get_lexicon([:AB, :BA, :CC], 50ms)          # every phoneme lasts 50 ms
seq = generate_sequence(word_phonemes_sequence; lexicon, presentations = 6, mode = :fixed,
                        weights = Dict(:AB => 1.0, :BA => 1.0, :CC => 1.0), seed = 1)
seq.sequence[seq.line_id.type, :]                      # :silence, :onset, :offset, ...
sign_intervals(:AB, seq)                               # [start, end] intervals in ms
sequence_end(seq)                                      # total duration in ms
```

## Stimuli for sequences

[`step_input`](@ref) creates one Poisson `MultiCompartmentStimulusGroup` per symbol, projecting
onto a random fraction of a dendritic population; [`update_stimuli!`](@ref) copies the intervals of
the sequence into the stimuli, and [`set_stimuli!`](@ref) switches word or phoneme stimuli on and
off. With the default `targets = [nothing]` the input targets a point-neuron population; pass the
compartments for dendritic neurons (with SNNUtils 0.2.9 and SNNModels 1.8.4 the default raised a
`MethodError`).

```julia
using SpikingNeuralNetworks, SNNUtils
SNN.@load_units
Exc = SNN.Tripod(N = 100, name = "Exc")
lexicon = get_lexicon([:AB, :BA], 50ms)
stim = step_input(; inputs = stimuli_names(lexicon).all, network = SNN.compose(; Exc),
                  pop = :Exc, targets = [:d1, :d2], p_post = 0.1, peak_rate = 8Hz,
                  proj_strength = 2.0)
seq = generate_sequence(word_phonemes_sequence; lexicon, presentations = 4, mode = :fixed,
                        weights = Dict(:AB => 1.0, :BA => 1.0))
model = SNN.compose(; Exc, stim)
update_stimuli!(; seq, model)                 # stimulus intervals from the sequence
set_stimuli!(; model, seq, phonemes = false)  # present words only
SNN.sim!(; model, duration = 200ms)
```

```@autodocs
Modules = [SNNUtils]
Pages   = ["SNNUtils.jl", "stimuli/sequence/stimuli.jl", "stimuli/sequence/sequence.jl",
           "stimuli/sequence/sequences/word_phonemes.jl"]
```

## Excitation/inhibition balance of dendritic neurons

[`compute_kei`](@ref), [`residual_current`](@ref) and [`optimal_kei`](@ref) compute the ratio of
inhibitory to excitatory input rate that cancels the net dendritic current of a dendritic neuron
with the soma held at a given potential; [`get_model`](@ref) computes the passive properties of
the neuron. `residual_current` is zero at the ratio returned by `compute_kei`. (In SNNUtils 0.2.9
these functions always threw, and the leak term of `residual_current` had the opposite sign.)
The non-exported helpers `SNNUtils.critical_window` and `SNNUtils.all_windows` estimate the
bimodality of membrane-potential distributions with the kernel-density method of Silverman
(1981).

```@autodocs
Modules = [SNNUtils]
Pages   = ["stimuli/balance_EI/compute_kei.jl", "stimuli/balance_EI/bimodal_kernel.jl"]
```

## BioSeq tasks

Import of artificial-grammar tasks in the BioSeq JSON format and storage of the network activity
for external analysis: [`import_bioseq_tasks`](@ref), [`bioseq_epochs`](@ref),
[`bioseq_lexicon`](@ref), [`seq_bioseq`](@ref), and the writers [`root_path`](@ref),
[`store_experiment_data`](@ref), [`store_target_pops`](@ref), [`store_labels`](@ref),
[`store_activity_data`](@ref) (HDF5 through DrWatson, NPZ for membrane traces). These functions
expect populations named `E`, `I1`, `I2` and specific JSON fields. (In SNNUtils 0.2.9
`import_bioseq_tasks` read the files from swapped folders, the inhibitory neuron ranges of
`store_experiment_data` were labelled in the wrong order, and `store_activity_data` threw.)

```@autodocs
Modules = [SNNUtils]
Pages   = ["stimuli/bioseq/import_bioseq.jl"]
```

## Weights and decoding analysis

- [`average_weight_dynamics`](@ref): mean weight between two neuron groups over a weight record.
- [`spikecount_features`](@ref), [`sym_features`](@ref): feature matrices (neurons x windows) from
  spike counts or recorded variables.
- [`SVCtrain`](@ref) (linear SVM) and `SNNUtils.LogRegtrain` (multinomial logistic regression):
  train/test a decoder, return Cohen's kappa and the confusion matrix.
  [`MultinomialLogisticRegression`](@ref): random train/test split, returns the test accuracy
  (it always threw in 0.2.9).
- [`score_spikes`](@ref): decode the presented word from the activity of the word assemblies.
- [`trial_average`](@ref), [`trial_sort`](@ref), [`symbols_to_int`](@ref), [`standardize`](@ref),
  [`do_pca`](@ref): helpers. (The undefined `pca` was exported up to 0.2.9.)

```julia
using SpikingNeuralNetworks, SNNUtils
SNN.@load_units
E = SNN.Poisson(N = 20, param = SNN.PoissonParameter(10Hz))
SNN.monitor!(E, [:fire])
SNN.sim!([E]; duration = 2s)
windows = [[t, t + 100ms] for t in 0ms:100ms:1900ms]
X = spikecount_features(E, windows)                    # 20 x 20 spike counts
labels = repeat([:a, :b], 10)
X[1:5, labels .== :b] .+= 5                            # make the two classes separable
kappa, cm = SVCtrain(X, labels)
code, ulabels = trial_average(X, labels)               # mean features per label
```

```@autodocs
Modules = [SNNUtils]
Pages   = ["analysis/weights.jl", "analysis/classifiers.jl"]
```
