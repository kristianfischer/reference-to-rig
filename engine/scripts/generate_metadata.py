#!/usr/bin/env python
"""Generate metadata.json from NAM files in capture_library."""

import json
import re
from pathlib import Path


def generate_metadata():
    """Scan NAM files and generate metadata.json."""
    nam_dir = Path("capture_library/nam_models")
    ir_dir = Path("capture_library/cab_irs")
    
    nam_files = list(nam_dir.glob("*.nam"))
    ir_files = list(ir_dir.glob("*.wav"))
    
    # Filter out placeholder files (< 1KB)
    real_nams = [f for f in nam_files if f.stat().st_size > 1000]
    real_irs = [f for f in ir_files if f.stat().st_size > 1000]
    
    print(f"Found {len(real_nams)} real NAM files")
    print(f"Found {len(real_irs)} real IR files")
    
    # Process NAM files
    models = []
    for f in real_nams:
        name = f.stem
        
        # Create a clean ID
        clean_id = re.sub(r"[^a-zA-Z0-9]", "_", name).lower()
        clean_id = re.sub(r"_+", "_", clean_id).strip("_")[:50]
        
        # Determine style/gain from filename
        name_upper = name.upper()
        if "HI" in name_upper and "HIGH" not in name_upper:
            gain_range = "high_gain"
            style = "high_gain"
        elif "LO" in name_upper:
            gain_range = "crunch"
            style = "crunch"
        elif "CLEAN" in name_upper:
            gain_range = "clean"
            style = "clean"
        else:
            gain_range = "crunch"
            style = "crunch"
        
        # Brightness hint
        if "REVYHI" in name_upper or "BRIGHT" in name_upper:
            brightness = "bright"
        elif "DARK" in name_upper:
            brightness = "dark"
        else:
            brightness = "neutral"
        
        # Tags from filename
        tags = []
        
        # Amp brand detection
        if "FENDER" in name_upper or "F.PLEX" in name_upper or "TWIN" in name_upper:
            tags.extend(["fender", "american"])
        if "MARSHALL" in name_upper or "JCM" in name_upper or "PLEXI" in name_upper:
            tags.extend(["marshall", "british"])
        if "MESA" in name_upper or "RECTIFIER" in name_upper or "BOOGIE" in name_upper:
            tags.extend(["mesa", "high_gain"])
        if "VOX" in name_upper or "AC30" in name_upper:
            tags.extend(["vox", "british", "chimey"])
        if "DUMBLE" in name_upper:
            tags.extend(["dumble", "boutique"])
        if "5150" in name_upper or "EVH" in name_upper:
            tags.extend(["evh", "high_gain"])
        
        # Capture type
        if "DI" in name_upper:
            tags.append("di")
        if "BLEND" in name_upper:
            tags.append("blend")
        if "SM57" in name_upper:
            tags.append("sm57")
        if "SM58" in name_upper:
            tags.append("sm58")
        if "RIBBON" in name_upper or "R121" in name_upper:
            tags.append("ribbon")
        
        # Gain tags
        tags.append(style)
        
        # Remove duplicates
        tags = list(set(tags))
        
        models.append({
            "id": clean_id,
            "name": name,
            "file_path": f"nam_models/{f.name}",
            "capture_type": "nam_model",
            "style": style,
            "gain_range": gain_range,
            "brightness": brightness,
            "tags": tags
        })
    
    # Process IR files
    irs = []
    for f in real_irs:
        name = f.stem
        clean_id = re.sub(r"[^a-zA-Z0-9]", "_", name).lower()
        clean_id = re.sub(r"_+", "_", clean_id).strip("_")[:50]
        
        # Determine brightness from filename
        name_upper = name.upper()
        if "BRIGHT" in name_upper or "JENSEN" in name_upper:
            brightness = "bright"
        elif "DARK" in name_upper:
            brightness = "dark"
        else:
            brightness = "neutral"
        
        # Tags
        tags = []
        if "V30" in name_upper or "VINTAGE30" in name_upper:
            tags.append("v30")
        if "GREENBACK" in name_upper:
            tags.append("greenback")
        if "4X12" in name_upper:
            tags.append("4x12")
        if "2X12" in name_upper:
            tags.append("2x12")
        if "1X12" in name_upper:
            tags.append("1x12")
        
        irs.append({
            "id": clean_id,
            "name": name,
            "file_path": f"cab_irs/{f.name}",
            "capture_type": "cab_ir",
            "style": "neutral",
            "brightness": brightness,
            "tags": tags
        })
    
    # Create metadata
    metadata = {
        "nam_models": models,
        "cab_irs": irs
    }
    
    # Save
    output_path = Path("capture_library/metadata.json")
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved metadata to {output_path}")
    print(f"  - {len(models)} NAM models")
    print(f"  - {len(irs)} Cabinet IRs")
    
    # Show first few
    if models:
        print("\nFirst 3 NAM models:")
        for m in models[:3]:
            print(f"  {m['id']}: {m['style']}, {m['brightness']}")


if __name__ == "__main__":
    generate_metadata()


