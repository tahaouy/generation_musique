# chopin-music-generation

A full symbolic music generation pipeline focused on Chopin-only MIDI data.
The model is trained end-to-end from raw MIDI notes and generates new piano sequences.
I already tried to make one in 2024 but since I got results that weren't very convincing, I decided to put it on hold. Now that I've gained a lot more knowledge about AI and deep learning, I gave it another try and the outcome is much better

Generation backend: TensorFlow/Keras GRU model (local training, GPU optional).

## Architecture

```
MIDI files (midi_chopin/)
        |
    data.py             <- load MIDI, extract note/chord tokens, build vocabulary
        |
    sequence builder    <- create fixed-length token windows and next-token targets
        |
    model.py            <- Embedding + stacked GRU + softmax classifier
        |
    callbacks           <- checkpoint, early stopping, learning-rate reduction
        |
    generate.py         <- temperature sampling + MIDI rendering
        |
    main.py             <- full train + generate orchestration via CLI
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you had previous package conflicts, recreate the environment from scratch:

```bash
rmdir /s /q .venv
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

**Step 1 - verify dataset is present in `midi_chopin/`**

**Step 2 - train and generate:**
```bash
python main.py
```

**Step 3 - custom run (recommended for showcase):**
```bash
python main.py --midi-folder midi_chopin --sequence-length 64 --epochs 80 --batch-size 128 --generated-length 400 --temperature 0.8 --output-midi generated_chopin.mid
```

## Generated sample (audio)

The model output was rendered to **[`generated_chopin.mp3`](generated_chopin.mp3)** (same training/generation settings as the “custom run” command above) so you can listen in the browser on GitHub.

<audio controls src="https://raw.githubusercontent.com/tahaouy/generation_musique/main/generated_chopin.mp3">
  Your browser does not support embedded audio — use the download link below.
</audio>

**Direct links:** [open the file in the repo](generated_chopin.mp3) · [download (raw)](https://raw.githubusercontent.com/tahaouy/generation_musique/main/generated_chopin.mp3)

> **Note:** `python main.py … --output-midi generated_chopin.mid` still writes a MIDI file locally; that file is ignored by Git (see `.gitignore`). The MP3 in the repo is the showcase listenable export.

## Design decisions

**Why Chopin-only data?** A single-composer corpus improves stylistic coherence and reduces cross-artist noise in generated motifs.

**Why integer targets instead of one-hot?** Sparse targets reduce memory usage and speed up training while keeping the same objective.

**Why Embedding + GRU?** Embeddings learn token relationships directly, and GRUs are efficient for sequence modeling with fewer parameters than larger recurrent stacks.

**Why temperature sampling?** Greedy decoding is repetitive; temperature gives controllable diversity for more musical outputs.

**Why training callbacks?** Early stopping and learning-rate scheduling stabilize convergence and avoid overtraining on small symbolic datasets.

## Stack

Python · TensorFlow/Keras · NumPy · music21 · MIDI symbolic modeling
