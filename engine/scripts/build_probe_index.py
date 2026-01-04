#!/usr/bin/env python
"""
Build FAISS index from NAM models using probe-based acoustic signatures.

This script:
1. Loads DI probe signals from capture_library/probes/
2. Renders each probe through each NAM model
3. Extracts 155-dim acoustic features from rendered audio
4. Aggregates features across probes to create NAM "signatures"
5. Computes normalization statistics across the library
6. Builds and saves FAISS index for fast retrieval

Usage:
    python scripts/build_probe_index.py

Prerequisites:
    - DI probe files in capture_library/probes/*.wav
    - NAM models in capture_library/nam_models/*.nam
    - metadata.json with NAM model listings

Output:
    - data/nam_acoustic_index.bin (FAISS index)
    - data/ir_acoustic_index.bin (FAISS index for IRs)
    - data/normalization_stats.json (z-score parameters)
    - data/index_metadata.json (model ID mapping)
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import structlog

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.features.acoustic_features import AcousticFeatureExtractor, FEATURE_DIM

logger = structlog.get_logger()

# Configuration
PROBE_DIR = Path("capture_library/probes")
NAM_DIR = Path("capture_library/nam_models")
IR_DIR = Path("capture_library/cab_irs")
OUTPUT_DIR = Path("data")
SAMPLE_RATE = 48000


def load_probes() -> list[tuple[str, np.ndarray]]:
    """
    Load all probe DI signals.
    
    Returns:
        List of (probe_name, audio_array) tuples
    """
    probes = []
    
    if not PROBE_DIR.exists():
        logger.warning(f"Probe directory not found: {PROBE_DIR}")
        logger.info("Creating probe directory. Please add DI probe WAV files.")
        PROBE_DIR.mkdir(parents=True, exist_ok=True)
        return probes
    
    probe_files = sorted(PROBE_DIR.glob("*.wav"))
    
    if not probe_files:
        logger.warning("No probe files found! Please add WAV files to capture_library/probes/")
        return probes
    
    logger.info(f"Loading {len(probe_files)} probe files...")
    
    for probe_path in probe_files:
        try:
            audio, sr = sf.read(probe_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            
            audio = audio.astype(np.float32)
            probes.append((probe_path.stem, audio))
            logger.debug(f"  Loaded: {probe_path.name} ({len(audio)/SAMPLE_RATE:.2f}s)")
            
        except Exception as e:
            logger.error(f"Failed to load probe {probe_path}: {e}")
    
    logger.info(f"Loaded {len(probes)} probes")
    return probes


def load_nam_model(model_path: Path):
    """
    Load a NAM model.
    
    Returns:
        NAM model object or None if failed
    """
    try:
        import json
        import torch
        from nam.models import init_from_nam
        
        with open(model_path, "r") as f:
            config = json.load(f)
        
        model = init_from_nam(config)
        model = model.eval()
        
        # Use CPU for batch processing
        model = model.to("cpu")
        
        return model
        
    except Exception as e:
        logger.debug(f"Failed to load NAM {model_path.name}: {e}")
        return None


def render_through_nam(model, audio: np.ndarray) -> np.ndarray:
    """
    Render audio through a NAM model.
    
    Args:
        model: NAM model
        audio: Input audio (mono, float32)
        
    Returns:
        Rendered audio
    """
    import torch
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    
    # Process
    with torch.no_grad():
        output = model(audio_tensor)
    
    return output.squeeze(0).numpy()


def build_nam_index(probes: list[tuple[str, np.ndarray]], extractor: AcousticFeatureExtractor) -> tuple:
    """
    Build acoustic index for NAM models.
    
    Args:
        probes: List of (name, audio) probe tuples
        extractor: Feature extractor
        
    Returns:
        (features_array, model_ids, model_names)
    """
    import faiss
    
    # Load metadata
    metadata_path = settings.capture_library_dir / "metadata.json"
    if not metadata_path.exists():
        logger.error("metadata.json not found! Run generate_metadata.py first.")
        return None, [], []
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    nam_models = metadata.get("nam_models", [])
    logger.info(f"Processing {len(nam_models)} NAM models...")
    
    all_features = []
    all_raw_features = []  # For normalization
    model_ids = []
    model_names = []
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    for i, model_info in enumerate(nam_models):
        model_id = model_info["id"]
        model_name = model_info["name"]
        file_path = settings.capture_library_dir / model_info["file_path"]
        
        if not file_path.exists():
            logger.debug(f"NAM file not found: {file_path}")
            failed += 1
            continue
        
        # Load model
        model = load_nam_model(file_path)
        if model is None:
            failed += 1
            continue
        
        # Render all probes and extract features
        probe_features = []
        
        for probe_name, probe_audio in probes:
            try:
                # Render probe through NAM
                rendered = render_through_nam(model, probe_audio)
                
                # Extract features (raw, no normalization yet)
                features = extractor.extract_raw(rendered)
                probe_features.append(features)
                
            except Exception as e:
                logger.debug(f"Failed to process probe {probe_name} through {model_name}: {e}")
        
        if not probe_features:
            logger.debug(f"No successful probe renders for {model_name}")
            failed += 1
            continue
        
        # Aggregate features across probes (mean)
        aggregated = np.mean(probe_features, axis=0)
        
        all_raw_features.append(aggregated)
        model_ids.append(model_id)
        model_names.append(model_name)
        successful += 1
        
        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == len(nam_models):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(nam_models) - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {i+1}/{len(nam_models)} "
                f"({successful} ok, {failed} failed) "
                f"- {remaining:.0f}s remaining"
            )
    
    if not all_raw_features:
        logger.error("No NAM models were successfully processed!")
        return None, [], []
    
    # Convert to array
    raw_features_array = np.array(all_raw_features, dtype=np.float32)
    
    # Compute normalization statistics
    logger.info("Computing normalization statistics...")
    extractor.compute_normalization_stats(raw_features_array)
    
    # Apply normalization and L2 normalize
    logger.info("Normalizing feature vectors...")
    normalized_features = (raw_features_array - extractor.norm_mean) / extractor.norm_std
    
    # L2 normalize each vector
    norms = np.linalg.norm(normalized_features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized_features = normalized_features / norms
    
    elapsed = time.time() - start_time
    logger.info(
        f"NAM index complete: {successful} models indexed in {elapsed:.1f}s "
        f"({failed} failed)"
    )
    
    return normalized_features.astype(np.float32), model_ids, model_names


def build_ir_index(extractor: AcousticFeatureExtractor) -> tuple:
    """
    Build acoustic index for cabinet IRs.
    
    IRs are characterized by their impulse response characteristics.
    """
    import faiss
    
    metadata_path = settings.capture_library_dir / "metadata.json"
    if not metadata_path.exists():
        return None, [], []
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    cab_irs = metadata.get("cab_irs", [])
    if not cab_irs:
        logger.info("No cabinet IRs found in metadata")
        return None, [], []
    
    logger.info(f"Processing {len(cab_irs)} cabinet IRs...")
    
    all_features = []
    ir_ids = []
    ir_names = []
    
    for ir_info in cab_irs:
        ir_id = ir_info["id"]
        ir_name = ir_info["name"]
        file_path = settings.capture_library_dir / ir_info["file_path"]
        
        if not file_path.exists():
            logger.debug(f"IR file not found: {file_path}")
            continue
        
        try:
            # Load IR
            ir_audio, sr = sf.read(file_path)
            if len(ir_audio.shape) > 1:
                ir_audio = ir_audio[:, 0]  # Take first channel
            
            # Resample if needed
            if sr != SAMPLE_RATE:
                import librosa
                ir_audio = librosa.resample(ir_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            
            # For IRs, we extract features directly from the impulse
            # Pad to minimum length for feature extraction
            min_length = SAMPLE_RATE  # 1 second
            if len(ir_audio) < min_length:
                ir_audio = np.pad(ir_audio, (0, min_length - len(ir_audio)))
            
            features = extractor.extract_raw(ir_audio.astype(np.float32))
            all_features.append(features)
            ir_ids.append(ir_id)
            ir_names.append(ir_name)
            
        except Exception as e:
            logger.debug(f"Failed to process IR {ir_name}: {e}")
    
    if not all_features:
        return None, [], []
    
    # Normalize using same stats as NAM (if available)
    features_array = np.array(all_features, dtype=np.float32)
    
    if extractor.norm_mean is not None:
        features_array = (features_array - extractor.norm_mean) / extractor.norm_std
        norms = np.linalg.norm(features_array, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        features_array = features_array / norms
    
    logger.info(f"IR index complete: {len(ir_ids)} IRs indexed")
    
    return features_array.astype(np.float32), ir_ids, ir_names


def save_index(features: np.ndarray, index_path: Path) -> None:
    """Save FAISS index to disk."""
    import faiss
    
    # Create index (Inner Product for cosine similarity on normalized vectors)
    index = faiss.IndexFlatIP(FEATURE_DIM)
    index.add(features)
    
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index: {index_path} ({index.ntotal} vectors)")


def main():
    """Main entry point."""
    print("=" * 70)
    print("NAM ACOUSTIC INDEX BUILDER")
    print("=" * 70)
    print()
    
    # Ensure output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load probes
    print("[1/5] Loading DI probe signals...")
    probes = load_probes()
    
    if not probes:
        print()
        print("!" * 70)
        print("NO PROBE FILES FOUND!")
        print()
        print("Please add DI guitar WAV files to:")
        print(f"  {PROBE_DIR.absolute()}")
        print()
        print("Recommended probes:")
        print("  - low_e_sustain.wav (low E string, 3-4 sec)")
        print("  - a_note_sustain.wav (A note, 3-4 sec)")
        print("  - high_e_sustain.wav (high E 12th fret, 3-4 sec)")
        print("  - power_chord.wav (E5 power chord, 3 sec)")
        print("  - palm_mutes.wav (low E palm muted 8ths, 4 sec)")
        print("  - open_chord.wav (open E or G major, 3 sec)")
        print("!" * 70)
        return 1
    
    # Initialize feature extractor
    print(f"\n[2/5] Initializing feature extractor ({FEATURE_DIM} dimensions)...")
    extractor = AcousticFeatureExtractor(sample_rate=SAMPLE_RATE)
    
    # Build NAM index
    print(f"\n[3/5] Building NAM acoustic index...")
    print("       (This may take 30-60 minutes for 300+ models)")
    print()
    
    nam_features, nam_ids, nam_names = build_nam_index(probes, extractor)
    
    if nam_features is None:
        print("Failed to build NAM index!")
        return 1
    
    # Build IR index
    print(f"\n[4/5] Building cabinet IR index...")
    ir_features, ir_ids, ir_names = build_ir_index(extractor)
    
    # Save everything
    print(f"\n[5/5] Saving indices and metadata...")
    
    # Save NAM index
    save_index(nam_features, OUTPUT_DIR / "nam_acoustic_index.bin")
    
    # Save IR index (if we have IRs)
    if ir_features is not None and len(ir_features) > 0:
        save_index(ir_features, OUTPUT_DIR / "ir_acoustic_index.bin")
    
    # Save normalization stats
    extractor.save_normalization_stats(OUTPUT_DIR / "normalization_stats.json")
    
    # Save index metadata (ID mapping)
    index_metadata = {
        "feature_dim": FEATURE_DIM,
        "sample_rate": SAMPLE_RATE,
        "n_probes": len(probes),
        "probe_names": [p[0] for p in probes],
        "nam_models": [
            {"index": i, "id": nam_ids[i], "name": nam_names[i]}
            for i in range(len(nam_ids))
        ],
        "cab_irs": [
            {"index": i, "id": ir_ids[i], "name": ir_names[i]}
            for i in range(len(ir_ids))
        ] if ir_ids else [],
    }
    
    with open(OUTPUT_DIR / "index_metadata.json", "w") as f:
        json.dump(index_metadata, f, indent=2)
    
    print()
    print("=" * 70)
    print("INDEX BUILD COMPLETE!")
    print("=" * 70)
    print()
    print(f"  NAM models indexed: {len(nam_ids)}")
    print(f"  Cabinet IRs indexed: {len(ir_ids) if ir_ids else 0}")
    print(f"  Feature dimensions: {FEATURE_DIM}")
    print(f"  Probes used: {len(probes)}")
    print()
    print("Output files:")
    print(f"  {OUTPUT_DIR / 'nam_acoustic_index.bin'}")
    print(f"  {OUTPUT_DIR / 'ir_acoustic_index.bin'}")
    print(f"  {OUTPUT_DIR / 'normalization_stats.json'}")
    print(f"  {OUTPUT_DIR / 'index_metadata.json'}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

