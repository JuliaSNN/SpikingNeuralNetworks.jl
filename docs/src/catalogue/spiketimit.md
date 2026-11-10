# SNNUtils: Spike TIMIT speech inputs

```@meta
CurrentModule = SNNUtils.SpikeTimit
```

`SNNUtils.SpikeTimit` reads the Spike TIMIT corpus and turns real speech into inputs for
SNNModels networks. Load it with

```julia
using SNNUtils
using SNNUtils.SpikeTimit      # its names are not exported into SNNUtils
using MAT                      # only needed to read BAE spike files
```

## What Spike TIMIT is

Spike TIMIT is the TIMIT speech corpus converted into spike trains by auditory encoders. Every
sentence comes with a word transcription, a phone transcription and spike times.

!!! warning "Licence"
    The audio and transcriptions come from TIMIT (LDC93S1), which is licensed by the Linguistic
    Data Consortium and cannot be redistributed. Neither SNNUtils nor these docs contain corpus
    data. You need your own licensed copy of the corpus and of the spike files generated from it.
    The tests of SNNUtils use a small invented corpus with the same file formats.

SpikeTimit provides:

- readers of the transcriptions and of the spike files ([`create_dataset`](@ref),
  [`get_spiketimes`](@ref));
- selection of words and sentences ([`find_word`](@ref), [`select_words`](@ref));
- processing of spikes ([`resample_spikes!`](@ref), [`transform_into_bursts!`](@ref),
  [`select_frequencies!`](@ref), [`remove_frequencies!`](@ref));
- stacking of utterances into long inputs ([`get_ordered_spikes`](@ref));
- conversion to SNNModels stimuli and SNNUtils sequences ([`spiketime_parameter`](@ref),
  [`to_sequence`](@ref), [`label_intervals`](@ref)).

## Dataset layout and `SPIKETIMIT_PATH`

The corpus location is the argument `dir` of the functions, or, if it is not given, the directory
named by the environment variable `SPIKETIMIT_PATH`:

```bash
export SPIKETIMIT_PATH=/data/SpikeTIMIT
```

```julia
dataset_path()                   # the root: ENV["SPIKETIMIT_PATH"]
dataset_path("train")            # root/train
dataset_path("DOC", "TIMITDIC.TXT")
```

A missing variable gives an `ArgumentError` that names it. The expected layout (lowercase names):

```
SpikeTIMIT/
  DOC/TIMITDIC.TXT                         pronunciation dictionary
  train/dr<k>/<speaker>/<senID>.wav        marks a sentence (the audio is not read by Julia)
  train/dr<k>/<speaker>/<senID>.wrd        words:  "<t0> <t1> <word>"  (samples at 16 kHz)
  train/dr<k>/<speaker>/<senID>.phn        phones: "<t0> <t1> <phone>"
  train/dr<k>/<speaker>/<senID>_spike_timing.mat    BAE spikes
  train/dr<k>/<speaker>/<senID>_cochlea35.npz       cochlea spikes, 35 channels
  train/dr<k>/<speaker>/<senID>_cochlea70.npz       cochlea spikes, 70 channels
  test/...                                 same structure
```

`k` is the dialect region (1 to 8); the first letter of the speaker name is the gender (`f` or `m`).
[`create_dataset`](@ref) walks `train/` or `test/` and returns a `DataFrame` with one row per
sentence (columns `speaker`, `senID`, `dialect`, `gender`, `path`, `words`, `phones`, `sentence`);
`words` and `phones` are [`Transcription`](@ref)s. Spike files are read only when asked for.

## Encodings

| Encoding | File | Channels | Needs |
|----------|------|----------|-------|
| [`BAE`](@ref)`()` | `<path>_spike_timing.mat` | 620 (20 bands x 31 threshold neurons) | `using MAT` |
| [`Cochlea`](@ref)`(35)`, [`Cochlea`](@ref)`(70)` | `<path>_cochlea35.npz`, `<path>_cochlea70.npz` | 35, 70 | nothing |

The BAE files (threshold encoding of Pan et al., 2020) are read through a package extension: without
`using MAT`, reading them throws an error that says so. The raw BAE times are multiplied by
`SpikeTimitConfig().correction` (5) to give seconds. A cochlea file holds one array `arr_0` of
size 2 x K: row 1 the spike times, row 2 the 0-based channel of each spike; the times are taken
as seconds.

The strings of the old API (`"bae"`, `"cochlea35"`, `"cochlea70"`) are still accepted by
[`encoding`](@ref).

## Units

| Group | Functions | Type and unit |
|-------|-----------|---------------|
| Data functions | [`create_dataset`](@ref), [`get_spiketimes`](@ref), [`select_words`](@ref), [`get_ordered_spikes`](@ref), [`stack_spiketimes`](@ref), [`word_intervals`](@ref), processing functions, `silence_time`, `shift` | `Float64`, seconds |
| Conversion functions | [`to_snn_spiketimes`](@ref), [`spiketime_parameter`](@ref), [`ms_intervals`](@ref), [`label_intervals`](@ref), [`to_sequence`](@ref) | `Float32`, milliseconds (SNNModels convention) |
| Time-step functions | [`transcriptions_dt`](@ref), [`ft_dt`](@ref) | `Int` steps; `dt` in ms |

Transcription times are the sample indices of the files divided by the sample rate of
[`SpikeTimitConfig`](@ref) (16 kHz). Convert only at the boundary to SNNModels: a spike train in
seconds passed to a population in ms would run 1000 times too fast.

## Workflow

```julia
using SNNUtils, SNNUtils.SpikeTimit, MAT
import SNNModels
using Random

rng = MersenneTwister(1)

# 1. the sentences of the corpus
train = create_dataset(dataset_path("train"))

# 2. utterances of two words, at most 10 sentences per word (without replacement)
inputs = select_words(train, ["that", "she"]; encoding = BAE(), samples = 10, rng)
resample_spikes!(inputs.spikes; n_feat = 8)            # BAE: 620 -> 80 channels
transform_into_bursts!(inputs.spikes; rng)

# 3. one long input: every utterance 5 times, random order, 100 ms of silence in between
ids = mix_inputs(rng, length(inputs.durations), 5)
ordered, trans = get_ordered_spikes(inputs; ids, silence_time = 0.1, shift = 0.0)   # seconds

# 4. a stimulus for an SNNModels population (Float32, ms)
E = SNNModels.IF(N = 200)
stim = SNNModels.SpikeTimeStimulus(E, :ge; param = spiketime_parameter(ordered),
                                   conn = (p = 0.1, μ = 2.0),
                                   N = get_encoding_dimension(inputs.spikes))
model = SNNModels.compose(; E, stim)

# 5. word and phone timing as an SNNUtils sequence (Float32, ms)
seq = to_sequence(trans)
SNNModels.sim!(; model, duration = SNNUtils.sequence_end(seq))
```

Pass `N` to `SpikeTimeStimulus` explicitly. By default it is the largest neuron index that fires,
which undercounts the input channels if the highest ones never fire.

[`select_words`](@ref) samples sentences without replacement; `replace = true` samples with
replacement. `get_ordered_spikes` needs either `ids` or `repetitions`.

## Relation to SNNUtils sequences

SNNUtils generates synthetic word and phoneme sequences ([`generate_sequence`](@ref SNNUtils.generate_sequence))
and locks stimuli to their intervals ([`sign_intervals`](@ref SNNUtils.sign_intervals),
[`update_stimuli!`](@ref SNNUtils.update_stimuli!)). SpikeTimit supplies the real counterpart:

- [`to_sequence`](@ref) builds a sequence with the fields of a synthetic one from the word and
  phone transcriptions of a stacked input, so `sign_intervals`, `all_intervals`,
  `merge_intervals`, `sequence_end`, `stimuli_names` and `update_stimuli!` work on real speech.
  Every word occurrence is separated from the previous one by a silence element, so two
  consecutive utterances of the same word are two presentations.
- [`label_intervals`](@ref)`(trans.words, "that")` gives the intervals of one label as
  `Vector{Vector{Float32}}` in ms, the format of `SNNModels.set_intervals!`.

```julia
trans.words                                    # Transcription{Float64}, seconds
label_intervals(trans.words, "that")           # [[t0, t1], ...] in ms
seq = to_sequence(trans)
SNNUtils.sign_intervals(:that, seq)            # the same intervals, from the sequence
```

A word spelled like a phone label (a word `ah` and a phone `ah`) is looked up as the phone by
`sign_intervals`; `to_sequence` warns once about such collisions.

## Reference

```@autodocs
Modules = [@__MODULE__]
Pages   = ["stimuli/spiketimit/SpikeTimit.jl", "stimuli/spiketimit/config.jl",
           "stimuli/spiketimit/types.jl", "stimuli/spiketimit/encodings.jl",
           "stimuli/spiketimit/dataset.jl", "stimuli/spiketimit/spikes_io.jl",
           "stimuli/spiketimit/process.jl", "stimuli/spiketimit/extract.jl",
           "stimuli/spiketimit/stack.jl", "stimuli/spiketimit/convert.jl"]
```
