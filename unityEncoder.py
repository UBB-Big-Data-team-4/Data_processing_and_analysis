import base64
import json
import os
from io import BytesIO
from pathlib import Path
from PIL import Image

# -- Configuration ------------------------------------------------------------
ROOT_DIR = "./data/unity"
OUTPUT_DIR = "./data/output"

ORIGINAL_SIZE = (1280, 720)
TARGET_SIZE = (640, 640)
# Calculate scaling factors for bboxes
SCALE_X = TARGET_SIZE[0] / ORIGINAL_SIZE[0]
SCALE_Y = TARGET_SIZE[1] / ORIGINAL_SIZE[1]

# Mapping folders to process
SOLO_FOLDERS = [""]
# -----------------------------------------------------------------------------

def encode_and_resize(image_path):
    """Resizes image to 640x640 and returns base64 string."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85) # JPEG used for efficiency in large files
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_sequences():
    root = Path(ROOT_DIR)
    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    # Dictionary to keep file handles open for the 25 possible occupancy counts (0-24)
    file_handles = {}

    print("Starting data restructuring...")

    for folder_name in SOLO_FOLDERS:
        solo_folder = root / folder_name
        if not solo_folder.exists():
            continue

        print(f"Processing {folder_name}...")
        
        # Sort sequences numerically
        sequences = sorted(
            [d for d in solo_folder.iterdir() if d.name.startswith("sequence.")],
            key=lambda x: int(x.name.split(".")[-1])
        )

        for seq in sequences:
            print(f"Processing sequence {seq.name}...")
            json_file = seq / "step0.frame_data.json"
            rgb_file = seq / "step0.camera.png"

            if not (json_file.exists() and rgb_file.exists()):
                continue

            try:
                # 1. Parse JSON and extract scaled bboxes
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                raw_values = data['captures'][0]['annotations'][0]['values']
                actual_count = len(raw_values)
                
                # Cap the count at 24 for the file naming convention
                file_idx = min(actual_count, 24)
                
                bbox_lines = []
                for val in raw_values:
                    # Scale coordinates to match 640x640 resize
                    new_x = val['origin'][0] * SCALE_X
                    new_y = val['origin'][1] * SCALE_Y
                    new_w = val['dimension'][0] * SCALE_X
                    new_h = val['dimension'][1] * SCALE_Y
                    bbox_lines.append(f"{new_x:.1f} {new_y:.1f} {new_w:.1f} {new_h:.1f}")

                # 2. Process Images
                rgb_b64 = encode_and_resize(rgb_file)

                # 3. Write to the specific occupancy file
                if file_idx not in file_handles:
                    file_handles[file_idx] = open(out_path / f"people_{file_idx}.txt", "a")
                
                target_file = file_handles[file_idx]
                
                # Line 1: RGB Image
                target_file.write(rgb_b64 + "\n")
                # Line 2+: Coordinates
                if bbox_lines:
                    target_file.write("\n".join(bbox_lines) + "\n")
                
            except Exception as e:
                print(f"Error processing {seq}: {e}")
                continue

    # Close all files
    for fh in file_handles.values():
        fh.close()
    
    print("Task Completed.")

if __name__ == "__main__":
    process_sequences()