# DI Probe Signals for NAM Acoustic Indexing

This directory contains clean DI (Direct Input) guitar recordings used to characterize NAM models acoustically.

## Purpose

Each NAM model is rendered with these probe signals to create an "acoustic signature" - a 155-dimensional feature vector that captures how the NAM actually SOUNDS.

This enables accurate tone matching: when you upload a reference guitar tone, we compare its acoustic features to the pre-computed signatures to find the most similar NAM models.

## Recording Requirements

### Format
- **Sample Rate:** 48kHz
- **Bit Depth:** 24-bit (or 32-bit float)
- **Format:** WAV (uncompressed)
- **Channels:** Mono
- **Length:** 2-4 seconds per clip

### Signal Chain
```
Guitar → Cable → Audio Interface → DAW

NO AMPS, NO PEDALS, NO PLUGINS
```

The recordings must be completely dry/clean. The NAM models will add all character.

### Peak Levels
- Normal clips: -12 to -6 dBFS peak
- Quiet clips (for level testing): -18 dBFS peak
- Hot clips (for saturation testing): -6 dBFS peak

## Recommended Probes

### Essential (Minimum 6 files)

| Filename | Description | Duration |
|----------|-------------|----------|
| `low_e_sustain.wav` | Open low E string, let ring | 3-4 sec |
| `a_note_sustain.wav` | A note (5th fret low E), sustain | 3-4 sec |
| `high_e_sustain.wav` | High E 12th fret, sustain | 3-4 sec |
| `power_chord.wav` | E5 power chord, aggressive pick | 3 sec |
| `palm_mutes.wav` | Low E palm muted 8th notes | 4 sec |
| `open_chord.wav` | Open E or G major, strum & ring | 3 sec |

### Extended (Better accuracy)

| Filename | Description | Duration |
|----------|-------------|----------|
| `single_note_dynamics.wav` | Same note: soft → medium → hard picks | 4 sec |
| `volume_swell.wav` | Note with volume knob rolled up | 3 sec |
| `palm_mutes_quiet.wav` | Same as palm_mutes at -18 dBFS | 4 sec |
| `palm_mutes_hot.wav` | Same as palm_mutes at -6 dBFS | 4 sec |
| `clean_arpeggios.wav` | Clean fingerpicked arpeggios | 4 sec |
| `staccato_riff.wav` | Short percussive notes | 3 sec |

### Why These Specific Probes?

- **Sustained notes** → Reveals how distortion evolves during decay
- **Power chords** → Tests intermodulation distortion, "chunk"
- **Palm mutes** → The #1 test for "tight" vs "loose" amp feel
- **Level variations** → Reveals compression curve, clean-up behavior
- **Dynamics** → Tests touch sensitivity, amp "feel"

## After Recording

1. Place all WAV files in this directory
2. Run the index builder:
   ```bash
   cd engine
   python scripts/build_probe_index.py
   ```
3. This will take 30-60 minutes for 300+ NAM models
4. The acoustic index will be saved to `data/nam_acoustic_index.bin`

## Tips for Good Probes

1. **Tune your guitar** - out-of-tune harmonics will skew results
2. **Fresh strings recommended** - consistent harmonic content
3. **Bridge pickup** - most common for amp testing, consistent response
4. **Avoid noise** - record in a quiet environment
5. **No processing** - ensure your DAW track has no plugins
6. **Check levels** - peaks should be -12 to -6 dBFS for normal clips

## Troubleshooting

### "No probe files found"
Add WAV files to this directory and re-run the index builder.

### Poor matching results
- Ensure probes are truly dry (no amp sim bleeding through)
- Try adding more probe variety (different notes, articulations)
- Check that probe levels are appropriate (-12 to -6 dBFS)

### Index build is slow
This is normal. Each NAM model must render all probes. For 300+ models with 8 probes, this is ~2400 audio renders. On CPU, expect 30-60 minutes.

