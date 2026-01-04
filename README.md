# Reference-to-Rig

**Local guitar tone matching and rig recipe generator**

Transform any reference audio into a reproducible guitar tone "rig recipe" using Neural Amp Modeler (NAM) captures and cabinet IRs.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          REFERENCE-TO-RIG ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────────┐         HTTP/REST         ┌────────────────────────┐ │
│   │   Desktop UI     │◄────────────────────────►│    Audio Engine        │ │
│   │   (Tauri+React)  │       localhost:8000      │    (FastAPI)           │ │
│   │                  │                           │                        │ │
│   │  • Import Audio  │                           │  ┌──────────────────┐  │ │
│   │  • Isolate       │                           │  │  /isolation      │  │ │
│   │  • Match Tone    │                           │  │  SAM Audio Adapter│  │ │
│   │  • A/B Preview   │                           │  └──────────────────┘  │ │
│   │  • Export        │                           │  ┌──────────────────┐  │ │
│   └──────────────────┘                           │  │  /features       │  │ │
│                                                  │  │  Log-mel, STFT   │  │ │
│   ┌──────────────────────────────────────────┐  │  └──────────────────┘  │ │
│   │           Capture Library                 │  │  ┌──────────────────┐  │ │
│   │  ┌─────────────┐  ┌─────────────┐        │  │  │  /retrieval      │  │ │
│   │  │ nam_models/ │  │  cab_irs/   │        │  │  │  FAISS ANN       │  │ │
│   │  │  .nam files │  │  .wav IRs   │        │  │  └──────────────────┘  │ │
│   │  └─────────────┘  └─────────────┘        │  │  ┌──────────────────┐  │ │
│   │  ┌─────────────────────────────────────┐ │  │  │  /optimization   │  │ │
│   │  │ metadata.json + SQLite + FAISS idx  │ │  │  │  CMA-ES / Coord  │  │ │
│   │  └─────────────────────────────────────┘ │  │  └──────────────────┘  │ │
│   └──────────────────────────────────────────┘  │  ┌──────────────────┐  │ │
│                                                  │  │  /rendering      │  │ │
│                                                  │  │  NAM + IR + EQ   │  │ │
│                                                  │  └──────────────────┘  │ │
│                                                  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Audio Import**: Load WAV/MP3 files or record from audio interface
- **Guitar Isolation**: Separate guitar from mixed tracks using SAM Audio
- **Tone Matching**: Find best-matching NAM captures and cabinet IRs
- **Smart Suggestions**: Get 3 flavors - Balanced, Brighter, Thicker
- **EQ Optimization**: Auto-generated parametric EQ curves
- **A/B Preview**: Compare reference vs synthesized match
- **Export**: JSON preset + human-readable Markdown recipe

## Why Tauri + React?

We chose Tauri over Electron for the desktop UI:
- **10x smaller** bundle size (~15MB vs 150MB+)
- **Lower memory** footprint (Rust backend)
- **Better security** with Rust's memory safety
- **Native performance** with system webview
- **Cross-platform** (Windows, macOS, Linux)

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Rust (for Tauri)
- CUDA-capable GPU (recommended for SAM Audio)

### SAM Audio Setup (Guitar Isolation)

Reference-to-Rig uses [SAM-Audio (facebook/sam-audio-large)](https://huggingface.co/facebook/sam-audio-large) for guitar isolation. This is a foundation model from Meta AI for isolating any sound using text prompts.

**Setup Steps:**

1. **Request Model Access**: Visit [facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large) and request access (requires HuggingFace account)

2. **Authenticate with HuggingFace**:
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   # Enter your HuggingFace access token
   ```

3. **Install SAM Audio**:
   ```bash
   pip install sam-audio torch torchaudio
   ```

4. **Enable in Config**: Set `RTR_ISOLATION_BACKEND=sam_audio` in your `.env` file

**Mock Backend**: If you don't have SAM Audio access, the app works with `RTR_ISOLATION_BACKEND=mock` (default), which uses simple bandpass filtering for testing.

### Setup

```bash
# Clone and setup
git clone <repo>
cd reference-to-rig

# Setup Python environment
cd engine
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# (Optional) Setup SAM Audio
huggingface-cli login

# Build capture library index
python -m scripts.build_index

# Run engine
uvicorn app.main:app --reload --port 8000

# In another terminal, run UI
cd ../ui
npm install
npm run tauri dev
```

### Using Make (Alternative)

```bash
make setup      # Install all dependencies
make index      # Build capture library index
make engine     # Start FastAPI server
make ui         # Start desktop UI
make all        # Run everything
```

## Project Structure

```
reference-to-rig/
├── engine/                 # Python FastAPI service
│   ├── app/
│   │   ├── api/           # REST endpoints
│   │   ├── isolation/     # SAM Audio adapter
│   │   ├── features/      # Feature extraction
│   │   ├── retrieval/     # FAISS search
│   │   ├── optimization/  # EQ parameter search
│   │   ├── rendering/     # Audio rendering
│   │   ├── storage/       # Project management
│   │   ├── tasks/         # Background task queue
│   │   └── observability/ # Logging, metrics
│   ├── scripts/           # CLI utilities
│   └── tests/             # Unit tests
├── ui/                     # Tauri + React desktop app
│   ├── src/               # React components
│   └── src-tauri/         # Tauri backend
├── capture_library/        # NAM models + IRs
│   ├── nam_models/
│   ├── cab_irs/
│   └── metadata.json
├── docs/                   # Documentation
└── scripts/                # Build/setup scripts
```

## API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/projects` | POST | Create new project |
| `/projects/{id}/import` | POST | Upload audio file |
| `/projects/{id}/isolate` | POST | Run guitar isolation |
| `/projects/{id}/match` | POST | Run tone matching |
| `/projects/{id}/results` | GET | Get match results |

See [API Documentation](docs/api.md) for full details.

## How It Works

1. **Preprocess**: Resample to 48kHz mono, loudness normalize
2. **Isolate**: Extract guitar stem using SAM Audio (or mock)
3. **Segment**: Select cleanest segments for matching
4. **Retrieve**: Embed reference, find top candidates via FAISS
5. **Optimize**: Search EQ/gain parameters to minimize perceptual distance
6. **Render**: Generate synthesized audio for A/B comparison

See [How Matching Works](docs/matching.md) for technical details.

## Adding Your Own Captures

See [Add Your Own Captures](docs/add_captures.md) for instructions on:
- Adding NAM model files
- Adding cabinet IR files
- Updating metadata
- Rebuilding the index

## Roadmap

### Phase 1 (Current MVP)
- [x] Core matching pipeline
- [x] Mock SAM Audio backend
- [x] Mock NAM backend
- [x] FAISS retrieval
- [x] Desktop UI
- [x] JSON/Markdown export

### Phase 2
- [ ] Real SAM Audio integration
- [ ] Real NAM model processing
- [ ] GPU acceleration
- [ ] Batch processing

### Phase 3
- [ ] DAW plugin (VST3/AU)
- [ ] Cloud sync for captures
- [ ] Community capture sharing
- [ ] Advanced A/B with synchronized playback

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

