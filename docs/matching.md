# How Matching Works

This document explains the technical details of the tone matching pipeline in Reference-to-Rig.

## Overview

The matching pipeline transforms a reference audio clip into a reproducible "rig recipe" through these stages:

```
Reference Audio
      │
      ▼
┌─────────────┐
│ Preprocess  │  Resample, normalize loudness
└─────────────┘
      │
      ▼
┌─────────────┐
│   Isolate   │  Extract guitar stem (SAM Audio)
└─────────────┘
      │
      ▼
┌─────────────┐
│   Extract   │  155-dim acoustic features
└─────────────┘
      │
      ▼
┌─────────────┐
│  Retrieve   │  FAISS cosine similarity search
└─────────────┘
      │
      ▼
┌─────────────┐
│  Optimize   │  EQ parameter search
└─────────────┘
      │
      ▼
┌─────────────┐
│   Render    │  Generate preview audio
└─────────────┘
      │
      ▼
   Rig Recipe
```

## The Acoustic Matching System

### Why Acoustic Matching?

Traditional approaches match based on **metadata** (style labels like "crunch", "high gain", etc.). This fails because:

1. Labels are subjective - one person's "crunch" is another's "high gain"
2. Filenames don't capture actual tonal character
3. Two NAMs with the same label can sound completely different

Our approach matches based on **how NAMs actually sound**:

1. Each NAM is "played" with standardized DI probe signals
2. The rendered audio is analyzed to extract acoustic features
3. These features become the NAM's "acoustic signature"
4. We compare your reference to these signatures

### The Index Building Process

```
DI Probes                 NAM Library
   │                          │
   ▼                          ▼
┌─────┐    ┌─────────────────────┐
│ WAV │───▶│ Render each probe   │
└─────┘    │ through each NAM    │
           └─────────────────────┘
                    │
                    ▼
           ┌───────────────────┐
           │ Extract 155-dim   │
           │ acoustic features │
           └───────────────────┘
                    │
                    ▼
           ┌───────────────────┐
           │ Aggregate across  │
           │ probes (mean)     │
           └───────────────────┘
                    │
                    ▼
           ┌───────────────────┐
           │ Build FAISS index │
           │ with z-score norm │
           └───────────────────┘
```

Run the index builder:
```bash
cd engine
python scripts/build_probe_index.py
```

## Stage 1: Preprocessing

**Goal:** Normalize audio for consistent analysis

**Steps:**
1. Resample to 48kHz (standard for pro audio)
2. Convert stereo to mono
3. Loudness normalize to -18 LUFS using ITU-R BS.1770

**Why this matters:** Different source recordings have wildly different loudness levels and sample rates. Normalizing ensures our feature extraction is consistent.

## Stage 2: Guitar Isolation

**Goal:** Separate the guitar from other instruments

**Technology:** [SAM-Audio (facebook/sam-audio-large)](https://huggingface.co/facebook/sam-audio-large)

SAM-Audio is a foundation model from Meta AI for audio source separation using:
- **Text prompting**: Describe what you want to isolate (e.g., "Electric guitar playing")
- **Visual prompting**: Use video masks to identify sound sources
- **Span prompting**: Specify time ranges where the sound occurs

For Reference-to-Rig, we use **text prompting** with guitar-related descriptions:
- "Electric guitar playing"
- "Distorted electric guitar"
- "Clean electric guitar"

**Configuration Options:**
```env
RTR_SAM_AUDIO_PROMPT=Electric guitar playing
RTR_SAM_AUDIO_USE_RERANKING=true  # Better quality
RTR_SAM_AUDIO_RERANKING_CANDIDATES=4
```

**Confidence Score:**
The isolation module provides a confidence score (0-1) based on:
- Energy ratio between target and residual
- Higher ratio = cleaner separation

**Mock Backend:**
For development/testing without SAM Audio access, a mock backend applies bandpass filtering (80Hz-5kHz) to simulate isolation. Set `RTR_ISOLATION_BACKEND=mock` in your config.

## Stage 3: Acoustic Feature Extraction (155 dimensions)

**Goal:** Convert audio into comparable acoustic signatures

The feature extractor captures multiple aspects of tonal character:

### Mel Spectrogram Features (128 dims)
- 64 mel bands × 2 statistics (mean, std)
- Captures the overall EQ curve and tonal balance
- Mean = average energy in each band (brightness, bass content)
- Std = energy variance (dynamic character)

### Spectral Shape Features (10 dims)
| Feature | What it captures |
|---------|------------------|
| Spectral Centroid | "Brightness" - center of spectral mass |
| Spectral Rolloff | Where high frequencies drop off |
| Spectral Flatness | Tonal vs. noisy character |
| Spectral Bandwidth | Frequency spread |
| Zero Crossing Rate | Texture, "grittiness" |

Each feature includes mean and std across time.

### Dynamics Features (10 dims)
| Feature | What it captures |
|---------|------------------|
| RMS (mean, std, max, 90th percentile) | Overall level and dynamics |
| Crest Factor | Peak/RMS ratio - compression amount |
| Integrated Loudness (LUFS) | Perceived loudness |
| Onset Strength | Transient intensity |
| Dynamic Range | Loud vs. quiet parts |
| Attack Slope | How fast transients rise |

### Band Ratio Features (6 dims)
Energy distribution across frequency bands tuned for guitar:

| Band | Frequency Range | What it captures |
|------|-----------------|------------------|
| Sub/Bass | 20-150 Hz | Low-end thump |
| Low-Mid | 150-400 Hz | Body, warmth |
| Mid | 400-1000 Hz | Presence, meat |
| Upper-Mid | 1000-2500 Hz | Cut, bite |
| Presence | 2500-6000 Hz | Attack definition |
| Air | 6000-20000 Hz | Sizzle, sparkle |

### Level-Dependent Coloration (1 dim)
- Measures how brightness changes with playing level
- High values = brighter when louder (typical of tube amps)
- Computed as correlation between RMS and spectral centroid

### Normalization
1. **Z-score normalization**: `(x - mean) / std` using library statistics
2. **L2 normalization**: Unit length for cosine similarity

## Stage 4: Candidate Retrieval

**Goal:** Find similar captures in the library

**Technology:** FAISS (Facebook AI Similarity Search)
- IndexFlatIP for inner product (equivalent to cosine similarity on L2-normalized vectors)
- O(n) search time but fast for local library sizes

**Process:**
1. Extract 155-dim features from isolated reference
2. Query NAM acoustic index → Top 10 NAM models by similarity
3. Query IR acoustic index → Top 10 IRs by similarity

**Why FAISS with Inner Product?**
- L2-normalized vectors + inner product = cosine similarity
- More stable than Euclidean distance for high-dimensional vectors
- Handles the "curse of dimensionality" better

## Stage 5: EQ Optimization

**Goal:** Find EQ settings that make the synthesized tone match the reference

### Optimization Method: Coordinate Descent

We optimize these parameters:
- Input gain: -12 to +12 dB
- 6-band parametric EQ:
  - Low shelf (80-200 Hz)
  - Low-mid peak (200-500 Hz)
  - Mid peak (500-1500 Hz)
  - High-mid peak (1500-3500 Hz)
  - Presence peak (3500-6000 Hz)
  - High shelf (6000-12000 Hz)
- Each band: frequency, gain (-12 to +12 dB), Q (0.5 to 4)

### Loss Function

```
L = w1 * spectral_distance
  + w2 * dynamics_distance
  + w3 * eq_penalty
  + w4 * gain_penalty
  + w5 * flavor_constraint
```

Where:
- `spectral_distance`: Euclidean distance in spectral feature space
- `dynamics_distance`: Distance in dynamics feature space
- `eq_penalty`: Σ|band_gain| / 50 (prefer subtle adjustments)
- `gain_penalty`: |input_gain| / 20 (prefer unity gain)
- `flavor_constraint`: Penalty for not meeting flavor target

### Three Flavors

1. **Balanced:** Minimizes overall loss without constraints
2. **Brighter:** Adds penalty if estimated brightness < reference
3. **Thicker:** Adds penalty if estimated low-end < reference

## Stage 6: Rendering

**Goal:** Generate preview audio for A/B comparison

### Signal Chain:
```
Isolated Guitar
      │
      ▼
┌─────────────┐
│ Input Gain  │  Apply gain adjustment
└─────────────┘
      │
      ▼
┌─────────────┐
│  NAM Model  │  Neural amp modeling
└─────────────┘
      │
      ▼
┌─────────────┐
│ Cabinet IR  │  Convolution with impulse response
└─────────────┘
      │
      ▼
┌─────────────┐
│     EQ      │  6-band parametric + HP/LP
└─────────────┘
      │
      ▼
┌─────────────┐
│  Loudness   │  Normalize to match reference
└─────────────┘
      │
      ▼
   WAV Output
```

## DI Probe Requirements

For the best matching accuracy, record these DI probes:

### Essential Probes (6 files minimum)

| Filename | Description | Duration |
|----------|-------------|----------|
| `low_e_sustain.wav` | Open low E string, let ring | 3-4 sec |
| `a_note_sustain.wav` | A note (5th fret low E), sustain | 3-4 sec |
| `high_e_sustain.wav` | High E 12th fret, sustain | 3-4 sec |
| `power_chord.wav` | E5 power chord, aggressive pick | 3 sec |
| `palm_mutes.wav` | Low E palm muted 8th notes | 4 sec |
| `open_chord.wav` | Open E or G major, strum & ring | 3 sec |

### Recording Requirements
- **Sample Rate:** 48kHz
- **Bit Depth:** 24-bit or 32-bit float
- **Format:** WAV (uncompressed)
- **Signal Chain:** Guitar → Cable → Interface → DAW (NO AMPS/EFFECTS)
- **Peak Level:** -12 to -6 dBFS

See `capture_library/probes/README.md` for detailed instructions.

## Packages Used

| Package | Purpose |
|---------|---------|
| `librosa` | Audio analysis, mel spectrograms, spectral features |
| `pyloudnorm` | ITU-R BS.1770 loudness measurement |
| `numpy` | Numerical computation |
| `scipy` | Signal processing, convolution |
| `faiss-cpu` | Efficient similarity search |
| `soundfile` | Audio I/O |
| `neural-amp-modeler` | NAM model loading and inference |
| `sam-audio` | Guitar isolation from mixes |

## Limitations

### What We Can't Do:
- Recover exact amp knob positions
- Match non-linear dynamics perfectly
- Separate overlapping instruments in dense mixes
- Account for room acoustics in the reference

### What We Can Do:
- Find captures with similar **acoustic** character
- Match based on how NAMs actually **sound**, not metadata
- Suggest EQ adjustments to get closer
- Provide a starting point that's much faster than manual trial-and-error

## Future Improvements

1. **More Probes:** Add dynamics-varying probes to capture touch sensitivity
2. **Perceptual Loss:** Use neural audio codec perceptual features
3. **CMA-ES Optimization:** More robust global optimization
4. **Real-time Preview:** Stream audio through the chain for live A/B
5. **Multi-capture Blending:** Mix multiple NAM models for hybrid tones
