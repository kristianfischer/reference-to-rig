#!/usr/bin/env python
"""Build FAISS index from NAM metadata.

This creates searchable embeddings based on NAM characteristics
(style, gain_range, brightness, tags) to enable meaningful matching.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Feature dimensions for embedding
# We encode: style(4), gain_range(4), brightness(3), common_tags(13) = 24 dimensions
STYLES = ["clean", "crunch", "overdrive", "high_gain"]
GAIN_RANGES = ["clean", "crunch", "high_gain", "extreme"]
BRIGHTNESS = ["dark", "neutral", "bright"]
COMMON_TAGS = [
    "fender", "marshall", "mesa", "vox", "dumble", "evh",
    "american", "british", "boutique",
    "di", "sm57", "ribbon", "blend"
]

EMBEDDING_DIM = len(STYLES) + len(GAIN_RANGES) + len(BRIGHTNESS) + len(COMMON_TAGS)


def encode_nam(metadata: dict) -> np.ndarray:
    """Encode NAM metadata into embedding vector."""
    embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    idx = 0
    
    # Style one-hot (with some spread for similar styles)
    style = metadata.get("style", "crunch")
    if style in STYLES:
        style_idx = STYLES.index(style)
        embedding[idx + style_idx] = 1.0
        # Add spread to adjacent styles
        if style_idx > 0:
            embedding[idx + style_idx - 1] = 0.3
        if style_idx < len(STYLES) - 1:
            embedding[idx + style_idx + 1] = 0.3
    idx += len(STYLES)
    
    # Gain range
    gain = metadata.get("gain_range", "crunch")
    if gain in GAIN_RANGES:
        gain_idx = GAIN_RANGES.index(gain)
        embedding[idx + gain_idx] = 1.0
        if gain_idx > 0:
            embedding[idx + gain_idx - 1] = 0.3
        if gain_idx < len(GAIN_RANGES) - 1:
            embedding[idx + gain_idx + 1] = 0.3
    idx += len(GAIN_RANGES)
    
    # Brightness
    bright = metadata.get("brightness", "neutral")
    if bright in BRIGHTNESS:
        bright_idx = BRIGHTNESS.index(bright)
        embedding[idx + bright_idx] = 1.0
        if bright_idx > 0:
            embedding[idx + bright_idx - 1] = 0.2
        if bright_idx < len(BRIGHTNESS) - 1:
            embedding[idx + bright_idx + 1] = 0.2
    idx += len(BRIGHTNESS)
    
    # Tags
    tags = metadata.get("tags", [])
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in COMMON_TAGS:
            tag_idx = COMMON_TAGS.index(tag_lower)
            embedding[idx + tag_idx] = 0.5
    
    # Add small random variation to break ties
    embedding += np.random.randn(EMBEDDING_DIM) * 0.01
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    
    return embedding


def encode_ir(metadata: dict) -> np.ndarray:
    """Encode IR metadata into embedding vector."""
    embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    idx = 0
    
    # IRs are neutral on style/gain, focus on brightness
    idx += len(STYLES)
    idx += len(GAIN_RANGES)
    
    # Brightness (more important for IRs)
    bright = metadata.get("brightness", "neutral")
    if bright in BRIGHTNESS:
        bright_idx = BRIGHTNESS.index(bright)
        embedding[idx + bright_idx] = 1.0
    idx += len(BRIGHTNESS)
    
    # Tags for IRs
    tags = metadata.get("tags", [])
    for tag in tags:
        tag_lower = tag.lower()
        # Map IR-specific tags
        if "4x12" in tag_lower:
            embedding[idx + 5] = 0.5  # High gain association
        elif "1x12" in tag_lower or "2x12" in tag_lower:
            embedding[idx + 0] = 0.3  # Clean/vintage association
        if "v30" in tag_lower:
            embedding[idx + 2] = 0.5  # Mesa association
        if "greenback" in tag_lower:
            embedding[idx + 1] = 0.5  # Marshall association
    
    # Add variation
    embedding += np.random.randn(EMBEDDING_DIM) * 0.01
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    
    return embedding


def build_indices():
    """Build FAISS indices from metadata."""
    import faiss
    
    # Load metadata
    metadata_path = Path("capture_library/metadata.json")
    if not metadata_path.exists():
        print("ERROR: metadata.json not found. Run generate_metadata.py first.")
        return
    
    with open(metadata_path) as f:
        data = json.load(f)
    
    nam_models = data.get("nam_models", [])
    cab_irs = data.get("cab_irs", [])
    
    print(f"Building index for {len(nam_models)} NAM models and {len(cab_irs)} IRs")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    
    # Build NAM embeddings
    if nam_models:
        np.random.seed(42)  # For reproducibility
        nam_embeddings = np.array([encode_nam(m) for m in nam_models], dtype=np.float32)
        
        # Create FAISS index (Inner Product = cosine similarity for normalized vectors)
        nam_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        nam_index.add(nam_embeddings)
        
        # Save index
        index_path = Path("data/nam_index.bin")
        index_path.parent.mkdir(exist_ok=True)
        faiss.write_index(nam_index, str(index_path))
        print(f"  Saved NAM index: {index_path} ({nam_index.ntotal} vectors)")
        
        # Show distribution
        styles = {}
        for m in nam_models:
            s = m.get("style", "unknown")
            styles[s] = styles.get(s, 0) + 1
        print(f"  Style distribution: {styles}")
    
    # Build IR embeddings
    if cab_irs:
        np.random.seed(43)
        ir_embeddings = np.array([encode_ir(m) for m in cab_irs], dtype=np.float32)
        
        ir_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        ir_index.add(ir_embeddings)
        
        index_path = Path("data/ir_index.bin")
        faiss.write_index(ir_index, str(index_path))
        print(f"  Saved IR index: {index_path} ({ir_index.ntotal} vectors)")
    
    print("\nIndex building complete!")
    print("\nTo test the index, run:")
    print("  python -c \"from app.retrieval.search import CaptureSearch; s = CaptureSearch(); import numpy as np; print(s.find_nam_models(np.random.randn(24).astype(np.float32), k=5))\"")


if __name__ == "__main__":
    build_indices()
