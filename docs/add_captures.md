# Adding Your Own Captures and IRs

This guide explains how to add your own Neural Amp Modeler (NAM) captures and cabinet impulse responses (IRs) to the Reference-to-Rig library.

## Capture Library Structure

```
capture_library/
├── metadata.json       # Index of all captures
├── nam_models/         # NAM model files
│   ├── my_amp_clean.nam
│   └── my_amp_high.nam
└── cab_irs/            # Cabinet IR files
    ├── my_cab_v30.wav
    └── my_cab_greenback.wav
```

## Adding NAM Models

### Step 1: Get NAM Files

NAM models are `.nam` files created with [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler).

Sources:
- Create your own using NAM Trainer
- Download from [ToneHunt](https://tonehunt.org/)
- Community shares on forums

### Step 2: Add to Directory

Copy `.nam` files to `capture_library/nam_models/`:

```bash
cp ~/Downloads/fender_deluxe_clean.nam capture_library/nam_models/
```

### Step 3: Update Metadata

Edit `capture_library/metadata.json` to add the new model:

```json
{
  "nam_models": [
    {
      "id": "fender_deluxe_clean",
      "name": "Fender Deluxe Reverb Clean",
      "file_path": "nam_models/fender_deluxe_clean.nam",
      "capture_type": "nam_model",
      "style": "clean",
      "gain_range": "clean",
      "brightness": "bright",
      "tags": ["fender", "clean", "american", "sparkle"]
    }
  ]
}
```

### Metadata Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique identifier (no spaces, lowercase) |
| `name` | Yes | Human-readable display name |
| `file_path` | Yes | Path relative to capture_library/ |
| `capture_type` | Yes | Always "nam_model" |
| `style` | No | clean, crunch, overdrive, high_gain |
| `gain_range` | No | clean, crunch, high_gain |
| `brightness` | No | dark, neutral, bright |
| `tags` | No | Array of searchable tags |

## Adding Cabinet IRs

### Step 1: Get IR Files

Cabinet IRs are audio files (WAV recommended) containing impulse response recordings.

Sources:
- Record your own cabinet with measurement software
- Download from IR packs (Ownhammer, ML Sound Lab, etc.)
- Free community IRs

### Requirements:
- Format: WAV (16/24-bit, any sample rate)
- Length: 50-200ms recommended
- Mono preferred (stereo will be summed)

### Step 2: Add to Directory

Copy `.wav` files to `capture_library/cab_irs/`:

```bash
cp ~/Downloads/v30_4x12_sm57.wav capture_library/cab_irs/
```

### Step 3: Update Metadata

```json
{
  "cab_irs": [
    {
      "id": "v30_4x12_sm57",
      "name": "V30 4x12 SM57 Cap",
      "file_path": "cab_irs/v30_4x12_sm57.wav",
      "capture_type": "cab_ir",
      "style": "modern",
      "brightness": "neutral",
      "tags": ["v30", "4x12", "sm57", "tight"]
    }
  ]
}
```

## Rebuilding the Index

After adding captures, rebuild the FAISS index:

```bash
cd engine
python -m scripts.build_index
```

This will:
1. Scan `capture_library/` for new files
2. Compute embeddings for each capture
3. Build FAISS indices for fast retrieval
4. Update SQLite database

### Expected Output:

```
2024-01-15 10:30:00 INFO Building capture library index library=./capture_library
2024-01-15 10:30:00 INFO Found NAM model name=fender_deluxe_clean
2024-01-15 10:30:00 INFO Found cabinet IR name=v30_4x12_sm57
2024-01-15 10:30:01 INFO Computing NAM embeddings...
2024-01-15 10:30:01 INFO Computing IR embeddings...
2024-01-15 10:30:02 INFO FAISS index saved path=./data/nam_index.bin vectors=10
2024-01-15 10:30:02 INFO FAISS index saved path=./data/ir_index.bin vectors=8
2024-01-15 10:30:02 INFO Metadata saved path=./capture_library/metadata.json
2024-01-15 10:30:02 INFO Index build complete nam_count=10 ir_count=8
```

## Best Practices

### Naming Conventions

Use descriptive, consistent names:
- `{brand}_{model}_{character}` for NAM models
- `{speaker}_{config}_{mic}` for IRs

Examples:
- `marshall_jcm800_crunch.nam`
- `greenback_4x12_r121.wav`

### Tagging Strategy

Use tags that describe:
- **Amp type:** fender, marshall, mesa, vox, boutique
- **Character:** clean, crunch, high_gain, lead, rhythm
- **Era:** vintage, modern, classic
- **Tone:** bright, dark, warm, aggressive, smooth

### IR Recommendations

For best matching results:
- Include IRs with different brightness levels
- Mix of cab sizes (1x12, 2x12, 4x12)
- Various speaker types (Greenback, V30, Jensen, Alnico)
- Different mic positions if available

### Embedding Quality

The matching system works better with:
- Diverse captures covering the tonal spectrum
- Accurate metadata (especially brightness/gain)
- Good quality source recordings

## Troubleshooting

### "NAM model not found"

Check that:
1. File exists in `nam_models/` directory
2. `file_path` in metadata is correct
3. Index was rebuilt after adding

### "IR sounds wrong"

Common issues:
- IR too short (< 30ms)
- Sample rate mismatch (will be resampled automatically)
- IR is actually just noise or silence

### "Matching doesn't find my new captures"

1. Verify metadata has correct `capture_type`
2. Check embedding was computed (look at index size)
3. Ensure tags/brightness/style are set for filtering

## Example: Adding a Complete Rig

Let's add a Marshall JCM800 + Greenback rig:

### 1. Copy Files

```bash
cp jcm800_master_vol.nam capture_library/nam_models/
cp greenback_4x12_center.wav capture_library/cab_irs/
```

### 2. Add Metadata

```json
{
  "nam_models": [
    {
      "id": "jcm800_master_vol",
      "name": "Marshall JCM800 2203",
      "file_path": "nam_models/jcm800_master_vol.nam",
      "capture_type": "nam_model",
      "style": "crunch",
      "gain_range": "crunch",
      "brightness": "neutral",
      "tags": ["marshall", "jcm800", "british", "crunch", "rock", "classic"]
    }
  ],
  "cab_irs": [
    {
      "id": "greenback_4x12_center",
      "name": "Celestion Greenback 4x12 Center",
      "file_path": "cab_irs/greenback_4x12_center.wav",
      "capture_type": "cab_ir",
      "style": "vintage",
      "brightness": "neutral",
      "tags": ["celestion", "greenback", "4x12", "british", "classic"]
    }
  ]
}
```

### 3. Rebuild Index

```bash
cd engine
python -m scripts.build_index
```

### 4. Test

Create a new project and run matching—your new captures should now appear in results!


