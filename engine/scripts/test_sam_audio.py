#!/usr/bin/env python
"""Test SAM Audio isolation backend."""

import sys
from pathlib import Path

# Add engine to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_sam_audio():
    """Test SAM Audio with a real audio file."""
    import numpy as np
    import soundfile as sf
    
    print("=" * 60)
    print("SAM AUDIO ISOLATION TEST")
    print("=" * 60)
    
    # Create test directory
    test_dir = Path("test_sam_audio")
    test_dir.mkdir(exist_ok=True)
    
    # Check for existing test audio or create one
    test_audio = test_dir / "test_mix.wav"
    
    if not test_audio.exists():
        print("\n[SETUP] Creating test audio mix...")
        # Create a test mix with "guitar" and "drums"
        sr = 48000
        duration = 5  # 5 seconds (shorter for faster testing)
        t = np.linspace(0, duration, int(sr * duration))
        
        # "Guitar" - E power chord with harmonics
        guitar = (
            0.4 * np.sin(2 * np.pi * 82.41 * t) +   # E2
            0.3 * np.sin(2 * np.pi * 123.47 * t) +  # B2
            0.3 * np.sin(2 * np.pi * 164.81 * t) +  # E3
            0.1 * np.sin(2 * np.pi * 329.63 * t)    # E4 harmonic
        )
        
        # Add some "distortion" character
        guitar = np.tanh(guitar * 2) * 0.5
        
        # "Drums" - kick and snare simulation
        drums = np.zeros_like(t)
        kick_times = np.arange(0, duration, 0.5)  # Kick every 0.5s
        snare_times = np.arange(0.25, duration, 0.5)  # Snare offset
        
        for kick_t in kick_times:
            idx = int(kick_t * sr)
            if idx < len(drums):
                decay = np.exp(-30 * np.linspace(0, 0.1, int(0.1 * sr)))
                end_idx = min(idx + len(decay), len(drums))
                drums[idx:end_idx] += 0.3 * np.sin(2 * np.pi * 60 * np.linspace(0, 0.1, end_idx - idx)) * decay[:end_idx - idx]
        
        for snare_t in snare_times:
            idx = int(snare_t * sr)
            if idx < len(drums):
                decay = np.exp(-50 * np.linspace(0, 0.05, int(0.05 * sr)))
                end_idx = min(idx + len(decay), len(drums))
                noise = np.random.randn(end_idx - idx) * 0.2
                drums[idx:end_idx] += noise * decay[:end_idx - idx]
        
        # Mix together
        mix = 0.6 * guitar + 0.4 * drums
        mix = mix / np.max(np.abs(mix)) * 0.8
        
        sf.write(test_audio, mix, sr)
        print(f"  Created: {test_audio}")
    else:
        print(f"\n[SETUP] Using existing test audio: {test_audio}")
    
    # Test SAM Audio backend
    print("\n[TEST] Loading SAM Audio backend...")
    
    try:
        from app.isolation.adapter import SAMAudioBackend
        
        # Initialize with small model + CPU for low VRAM systems
        # Model sizes: small (~1GB), base (~2GB), large (~4GB+ VRAM required)
        backend = SAMAudioBackend(
            model_name="facebook/sam-audio-small",  # Small model for low VRAM
            device="cpu",  # Force CPU to avoid OOM crashes
            prompt="electric guitar",
            use_reranking=True,
            reranking_candidates=1,  # Fast mode
        )
        
        print(f"  Backend: {backend.name}")
        print(f"  Device: {backend._device}")
        
        # Run isolation
        output_path = test_dir / "isolated_guitar.wav"
        
        print("\n[TEST] Running isolation...")
        print("  This may take a minute on first run (model download)...")
        
        def progress_callback(p):
            bar = "=" * int(p * 30)
            print(f"\r  Progress: [{bar:<30}] {p*100:.0f}%", end="", flush=True)
        
        result = backend.isolate(
            input_path=test_audio,
            output_path=output_path,
            progress_callback=progress_callback,
        )
        print()  # Newline after progress bar
        
        print("\n[RESULTS]")
        print(f"  Output: {output_path}")
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Sample Rate: {result['sample_rate']} Hz")
        print(f"  Prompt: {result['prompt_used']}")
        
        # Verify output exists
        if output_path.exists():
            audio, sr = sf.read(output_path)
            print(f"\n  Output verified: {len(audio)/sr:.2f}s @ {sr}Hz")
            print("\n[PASS] SAM Audio test PASSED!")
        else:
            print("\n[FAIL] Output file not created!")
            return False
            
        return True
        
    except ImportError as e:
        print(f"\n[FAIL] SAM Audio not installed: {e}")
        print("\nInstall with:")
        print("  pip install git+https://github.com/facebookresearch/sam-audio.git")
        return False
        
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_sam_audio()
    sys.exit(0 if success else 1)


